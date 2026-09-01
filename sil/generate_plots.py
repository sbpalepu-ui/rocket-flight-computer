import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rocket_sim import simulate_flight, synthesize_sensors
from kalman import run_filter
from fsm import run_fsm
from sil_test import run_nominal_suite, evaluate_run

OUT = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

# ---------- Figure 1: Kalman tracking (best/reference run) ----------
truth = simulate_flight(seed=1)
sensors = synthesize_sensors(truth, seed=1)
kf_out = run_filter(sensors)
fsm_out = run_fsm(kf_out)
imu_alt_integrated = None

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(truth["t"], truth["altitude"], color="black", lw=1.6, label="True altitude (ground truth)")
ax.plot(sensors["baro_t"], sensors["baro_alt"], ".", ms=2, alpha=0.35, color="tab:orange", label="Raw barometer")
ax.plot(kf_out["t"], kf_out["altitude"], color="tab:blue", lw=1.6, label="Kalman filter estimate")
ax.axvline(truth["apogee_t"], color="gray", ls="--", lw=1, label="True apogee")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Altitude (m)")
ax.set_title("Reference Flight: Kalman Estimate vs. Ground Truth vs. Raw Barometer")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_kalman_tracking.png", dpi=150); plt.close(fig)

# ---------- Figure 2: Phase transition timeline ----------
state_order = {"IDLE": 0, "BOOST": 1, "COAST": 2, "APOGEE": 3, "DESCENT": 4, "LANDED": 5}
state_nums = [state_order[s] for s in fsm_out["state"]]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.step(fsm_out["t"], state_nums, where="post", color="tab:blue", lw=1.5)
ax.set_yticks(list(state_order.values())); ax.set_yticklabels(list(state_order.keys()))
for ti, s in fsm_out["transitions"]:
    ax.axvline(ti, color="gray", ls=":", lw=0.8)
    ax.annotate(f"{s}\nt={ti:.2f}s", (ti, state_order[s]), textcoords="offset points",
                xytext=(4, 6), fontsize=7)
ax.set_xlabel("Time (s)"); ax.set_title("Flight Phase State Machine Timeline (Reference Flight)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_phase_timeline.png", dpi=150); plt.close(fig)

# ---------- Figure 3: Sensor dropout stress test ----------
truth_d = simulate_flight(seed=500)
sensors_d = synthesize_sensors(truth_d, dropout_window=(3.0, 5.0), seed=500)
kf_d = run_filter(sensors_d)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(truth_d["t"], truth_d["altitude"], color="black", lw=1.6, label="True altitude")
ax.plot(kf_d["t"], kf_d["altitude"], color="tab:blue", lw=1.6, label="Kalman estimate")
ax.axvspan(3.0, 5.0, color="red", alpha=0.15, label="Barometer dropout window")
ax.set_xlim(0, 12)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Altitude (m)")
ax.set_title("Stress Test: 2-Second Barometer Dropout During Coast")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_dropout_test.png", dpi=150); plt.close(fig)

# ---------- Figure 4: SIL 20-run apogee error distribution ----------
rows = run_nominal_suite(20)
errs = [r["apogee_err_ft"] for r in rows]
timing = [r["timing_err_s"] for r in rows]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(range(1, 21), errs, color="tab:blue")
axes[0].axhline(15, color="red", ls="--", lw=1, label="+-15 ft pass threshold")
axes[0].axhline(-15, color="red", ls="--", lw=1)
axes[0].set_xlabel("SIL run #"); axes[0].set_ylabel("Apogee error (ft)")
axes[0].set_title("Apogee Estimate Error — 20 Nominal Runs")
axes[0].legend(fontsize=8)

axes[1].bar(range(1, 21), timing, color="tab:green")
axes[1].axhline(0.5, color="red", ls="--", lw=1, label="+-0.5 s pass threshold")
axes[1].axhline(-0.5, color="red", ls="--", lw=1)
axes[1].set_xlabel("SIL run #"); axes[1].set_ylabel("Deploy timing error (s)")
axes[1].set_title("Deployment Signal Timing Error — 20 Nominal Runs")
axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_sil_results.png", dpi=150); plt.close(fig)

print("All figures saved to", OUT)
