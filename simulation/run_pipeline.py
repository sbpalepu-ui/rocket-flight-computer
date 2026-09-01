import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")

from rocket_sim import simulate_flight, synthesize_sensors, G
from kalman import run_filter
from fsm import run_fsm

truth = simulate_flight(seed=1)
sensors = synthesize_sensors(truth, seed=1)
kf_out = run_filter(sensors)
fsm_out = run_fsm(kf_out)

true_apogee_ft = truth["apogee_altitude_m"] * 3.28084
kf_apogee_m = np.max(kf_out["altitude"])
kf_apogee_ft = kf_apogee_m * 3.28084
apogee_err_ft = kf_apogee_ft - true_apogee_ft

print(f"True apogee:   {true_apogee_ft:.1f} ft at t={truth['apogee_t']:.2f}s")
print(f"Kalman apogee: {kf_apogee_ft:.1f} ft (error {apogee_err_ft:+.1f} ft)")
print(f"Deploy signal fired at t={fsm_out['deploy_t']}")
print("Transitions:")
for ti, s in fsm_out["transitions"]:
    print(f"  t={ti:6.2f}s  -> {s}")

# ---- Plot 1: Kalman filter tracking ----
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(truth["t"], truth["altitude"], label="True altitude", color="black", lw=1.5)
ax.plot(sensors["baro_t"], sensors["baro_alt"], ".", ms=2, alpha=0.3, label="Raw baro", color="tab:orange")
ax.plot(kf_out["t"], kf_out["altitude"], label="Kalman estimate", color="tab:blue", lw=1.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Altitude (m)")
ax.set_title("Kalman Filter Altitude Tracking vs. Ground Truth")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(DOCS_DIR, "kalman_tracking.png"), dpi=150)
print("Saved kalman_tracking.png")
