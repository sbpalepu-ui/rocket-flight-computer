"""
kalman.py
3-state constant-acceleration Kalman filter: x = [altitude, velocity, acceleration]

Runs a predict step every 1 ms (IMU rate) and an altitude-update step
whenever a barometer sample is available (50 Hz), plus an acceleration
update every IMU sample. This mirrors the multi-rate fusion described
in the project's Kalman_Task (100 Hz effective state output).
"""

import numpy as np


class AltitudeKalmanFilter:
    def __init__(self, q_accel_var=1.2, r_baro=0.25, r_imu=0.06):
        """
        q_accel_var: process noise power spectral density on jerk (accel rate
            of change). Higher = filter trusts the dynamics model less and
            reacts faster to sensor changes.
        r_baro: barometer measurement noise variance (m^2)
        r_imu: IMU measurement noise variance (g^2, in "g" units)
        """
        self.x = np.zeros(3)          # [alt, vel, accel]  (accel in m/s^2, gravity-removed)
        self.P = np.eye(3) * 10.0
        self.q_accel_var = q_accel_var
        self.R_baro = np.array([[r_baro]])
        self.R_imu = np.array([[r_imu * 9.81 ** 2]])
        self.H_baro = np.array([[1.0, 0.0, 0.0]])
        self.H_imu = np.array([[0.0, 0.0, 1.0]])
        self.initialized = False

    def initialize(self, alt0):
        self.x = np.array([alt0, 0.0, 0.0])
        self.P = np.diag([1.0, 4.0, 9.0])
        self.initialized = True

    def _F_Q(self, dt):
        F = np.array([
            [1, dt, 0.5 * dt ** 2],
            [0, 1, dt],
            [0, 0, 1],
        ])
        # Discretized white-noise-jerk process noise
        q = self.q_accel_var
        Q = q * np.array([
            [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
            [dt ** 4 / 8,  dt ** 3 / 3, dt ** 2 / 2],
            [dt ** 3 / 6,  dt ** 2 / 2, dt],
        ])
        return F, Q

    def predict(self, dt):
        if dt <= 0:
            return
        F, Q = self._F_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def _update(self, z, H, R):
        y = z - H @ self.x                       # residual
        S = H @ self.P @ H.T + R                  # residual covariance
        K = self.P @ H.T @ np.linalg.inv(S)       # Kalman gain
        self.x = self.x + (K @ y).flatten()
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P

    def update_baro(self, alt_measurement):
        if np.isnan(alt_measurement):
            return  # dropout: skip update, coast on prediction only
        self._update(np.array([alt_measurement]), self.H_baro, self.R_baro)

    def update_imu(self, accel_specific_force_ms2):
        # Remove gravity to get "world-frame" vertical acceleration
        accel_true = accel_specific_force_ms2 - 9.81
        self._update(np.array([accel_true]), self.H_imu, self.R_imu)


def run_filter(sensors, dt_output=0.01):
    """
    Fuse the full sensor stream through the Kalman filter and resample
    the output onto a fixed 100 Hz grid (matching the firmware Kalman_Task
    rate).
    """
    baro_t, baro_alt = sensors["baro_t"], sensors["baro_alt"]
    imu_t, imu_accel = sensors["imu_t"], sensors["imu_accel"]

    kf = AltitudeKalmanFilter()
    kf.initialize(alt0=np.nanmean(baro_alt[:10]))

    events = []
    for ti, alt in zip(baro_t, baro_alt):
        events.append((ti, "baro", alt))
    for ti, acc in zip(imu_t, imu_accel):
        events.append((ti, "imu", acc))
    events.sort(key=lambda e: e[0])

    out_t, out_alt, out_vel, out_acc = [], [], [], []
    last_t = events[0][0]
    next_output_t = last_t

    for ti, kind, val in events:
        kf.predict(ti - last_t)
        last_t = ti
        if kind == "baro":
            kf.update_baro(val)
        else:
            kf.update_imu(val)

        while ti >= next_output_t:
            out_t.append(next_output_t)
            out_alt.append(kf.x[0])
            out_vel.append(kf.x[1])
            out_acc.append(kf.x[2])
            next_output_t += dt_output

    return {
        "t": np.array(out_t), "altitude": np.array(out_alt),
        "velocity": np.array(out_vel), "acceleration": np.array(out_acc),
    }
