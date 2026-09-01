"""
rocket_sim.py
Vertical-axis (1-DOF translational) rocket flight simulator.

Produces:
  - Ground-truth trajectory (altitude, velocity, acceleration) at 1 ms resolution
  - Synthetic noisy barometer samples (50 Hz)
  - Synthetic noisy IMU accelerometer samples (1000 Hz, includes slow bias drift)

NOTE: The thrust curve below is a hand-modeled approximation of a typical
Estes E12 profile (peak ~20 N ignition spike, ~12 N average, ~2.4 s burn,
~30 N*s total impulse). It is NOT the vendor .eng file -- this sandbox has
no network access. For hardware-accurate work, replace THRUST_CURVE with
the real .eng data downloaded from thrustcurve.org.
"""

import numpy as np

# ----------------------------------------------------------------------
# Motor: approximate Estes E12-4 thrust curve (time_s, thrust_N)
# ----------------------------------------------------------------------
THRUST_CURVE = np.array([
    [0.00,  0.0],
    [0.02, 20.5],
    [0.06, 16.0],
    [0.10, 14.2],
    [0.30, 13.0],
    [0.60, 12.6],
    [1.00, 12.3],
    [1.40, 12.0],
    [1.80, 11.6],
    [2.10, 10.5],
    [2.30,  6.0],
    [2.42,  0.0],
])
BURN_TIME = THRUST_CURVE[-1, 0]
TOTAL_IMPULSE = np.trapezoid(THRUST_CURVE[:, 1], THRUST_CURVE[:, 0])  # N*s

# ----------------------------------------------------------------------
# Rocket physical parameters
# ----------------------------------------------------------------------
DIAMETER_M = 0.0508          # 2 in body tube
AREA_M2 = np.pi * (DIAMETER_M / 2) ** 2
CD = 0.52                    # drag coefficient (OpenRocket-typical for this airframe)
RHO = 1.225                  # kg/m^3, sea level
G = 9.81                     # m/s^2

DRY_MASS_KG = 0.360          # airframe + avionics, motor casing empty
PROPELLANT_MASS_KG = 0.0245  # E-class propellant mass
WET_MASS_KG = DRY_MASS_KG + PROPELLANT_MASS_KG

DESCENT_RATE_MPS = 5.2       # terminal velocity under 12" chute (approx)

DT = 0.001                   # 1 ms integration step


def thrust_at(t, scale=1.0):
    if t < 0 or t > BURN_TIME:
        return 0.0
    return scale * np.interp(t, THRUST_CURVE[:, 0], THRUST_CURVE[:, 1])


def mass_at(t):
    """Linear propellant depletion over the burn."""
    if t <= 0:
        return WET_MASS_KG
    if t >= BURN_TIME:
        return DRY_MASS_KG
    frac_burned = t / BURN_TIME
    return WET_MASS_KG - frac_burned * PROPELLANT_MASS_KG


def simulate_flight(thrust_scale=1.0, drag_scale=1.0, early_burnout_s=None,
                     t_max=60.0, seed=None):
    """
    Integrate vertical flight dynamics with RK4-equivalent fine-step Euler
    (dt = 1 ms, matching the firmware IMU sample interval).

    thrust_scale / drag_scale: run-to-run variation knobs used by the
        Monte Carlo SIL test harness (+/-5% thrust, +/-2% drag per the
        project's nominal-run test plan).
    early_burnout_s: if set, thrust is forced to zero after this time
        (simulates a motor CATO / early burnout test case).

    Returns a dict of ground-truth time series plus the true apogee.
    """
    rng = np.random.default_rng(seed)

    n_steps = int(t_max / DT) + 1
    t = np.zeros(n_steps)
    alt = np.zeros(n_steps)
    vel = np.zeros(n_steps)
    acc = np.zeros(n_steps)

    apogee_alt = None
    apogee_t = None
    landed_t = None
    phase = "BOOST"

    for i in range(1, n_steps):
        ti = t[i - 1]
        a_prev = alt[i - 1]
        v_prev = vel[i - 1]

        if early_burnout_s is not None and ti >= early_burnout_s:
            F_thrust = 0.0
        else:
            F_thrust = thrust_at(ti, scale=thrust_scale)

        m = mass_at(min(ti, BURN_TIME)) if apogee_t is None else DRY_MASS_KG

        if apogee_alt is None:
            # Powered / coasting ascent
            F_drag = 0.5 * RHO * v_prev * abs(v_prev) * CD * AREA_M2 * drag_scale
            F_net = F_thrust - F_drag - m * G
            a = F_net / m
            v = v_prev + a * DT
            a_new = a_prev + v * DT

            if v <= 0 and ti > 0.5:  # apogee reached (ignore the pad t=0 case)
                apogee_alt = a_prev
                apogee_t = ti
                phase = "DESCENT"
        elif landed_t is None:
            # Under chute: constant descent rate with a short transient
            v = -DESCENT_RATE_MPS
            a_new = max(a_prev + v * DT, 0.0)
            a = (v - v_prev) / DT
        else:
            # Already on the ground -- freeze state. (Bug found during
            # development: without this branch, velocity kept getting
            # reassigned to -DESCENT_RATE_MPS every step post-touchdown
            # even though altitude was clamped at 0, which fed a phantom
            # -5.2 m/s "sinking through the ground" velocity into the
            # sensor synthesis and made the Kalman filter's tail-end
            # estimate diverge to roughly -12 m/s instead of settling to 0.)
            v = 0.0
            a_new = 0.0
            a = 0.0

        if landed_t is None and apogee_alt is not None and a_new <= 0 and t[i-1] > apogee_t + 0.05:
            landed_t = ti

        t[i] = ti + DT
        alt[i] = a_new
        vel[i] = v
        acc[i] = a

        if landed_t is not None and ti > landed_t + 10.0:
            n_steps = i + 1
            break

    t, alt, vel, acc = t[:n_steps], alt[:n_steps], vel[:n_steps], acc[:n_steps]

    return {
        "t": t, "altitude": alt, "velocity": vel, "acceleration": acc,
        "apogee_altitude_m": apogee_alt, "apogee_t": apogee_t,
        "landed_t": landed_t, "burn_time": BURN_TIME,
    }


def synthesize_sensors(truth, baro_rate_hz=50, imu_rate_hz=1000,
                        baro_noise_std_m=0.5, imu_noise_std_g=0.05,
                        imu_bias_walk_std_g=0.00006, dropout_window=None,
                        noise_multiplier=1.0, seed=None):
    """
    Down-sample the ground truth to sensor rates and add realistic noise.

    dropout_window: optional (t_start, t_end) tuple during which
        barometer samples are withheld (NaN) -- used for the sensor
        dropout SIL test case.
    """
    rng = np.random.default_rng(seed)
    t, alt, acc = truth["t"], truth["altitude"], truth["acceleration"]

    # --- Barometer ---
    baro_dt = 1.0 / baro_rate_hz
    baro_t = np.arange(t[0], t[-1], baro_dt)
    baro_true = np.interp(baro_t, t, alt)
    baro_meas = baro_true + rng.normal(0, baro_noise_std_m * noise_multiplier, size=baro_t.shape)
    if dropout_window is not None:
        mask = (baro_t >= dropout_window[0]) & (baro_t <= dropout_window[1])
        baro_meas[mask] = np.nan

    # --- IMU accelerometer (reported in "sensed" accel = true accel + g, since
    #     an accelerometer at rest reads +1 g, not 0) ---
    imu_dt = 1.0 / imu_rate_hz
    imu_t = np.arange(t[0], t[-1], imu_dt)
    imu_true_accel = np.interp(imu_t, t, acc) + G   # convert to specific force
    bias_walk = np.cumsum(rng.normal(0, imu_bias_walk_std_g * G, size=imu_t.shape))
    imu_meas = imu_true_accel + bias_walk + rng.normal(
        0, imu_noise_std_g * G * noise_multiplier, size=imu_t.shape)

    return {
        "baro_t": baro_t, "baro_alt": baro_meas,
        "imu_t": imu_t, "imu_accel": imu_meas,
    }


if __name__ == "__main__":
    result = simulate_flight()
    print(f"Total impulse (approx E12 model): {TOTAL_IMPULSE:.1f} N*s")
    print(f"Burn time: {BURN_TIME:.2f} s")
    print(f"Apogee: {result['apogee_altitude_m']:.1f} m "
          f"({result['apogee_altitude_m']*3.28084:.0f} ft) at t={result['apogee_t']:.2f} s")
    print(f"Landed at t={result['landed_t']:.2f} s")
