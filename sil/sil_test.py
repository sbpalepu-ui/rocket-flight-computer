"""
sil_test.py
Software-in-the-loop (SIL) validation harness.

The project guide calls for Hardware-in-the-Loop (HIL) testing where the
Python simulation streams sensor packets over USB serial to a real ESP32
running the ported C++ firmware. Since this build stops at the simulation
stage (no physical hardware), this harness performs the equivalent
validation entirely in software: the *same* Kalman filter and FSM Python
modules that would be ported line-for-line to C++ are exercised against
the sim/sensor pipeline, using the exact test plan and pass criteria
from the project's HIL section (20 nominal runs + 3 stress cases).

This is a legitimate substitute for demonstrating the validation
*methodology* -- it does not substitute for the numerical confidence a
real HIL/flight test would provide, since it never leaves Python and
never touches a real ADC, SPI bus, or clock jitter. That caveat is
called out explicitly in the technical report.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation"))

import numpy as np
import csv

from rocket_sim import simulate_flight, synthesize_sensors, G
from kalman import run_filter
from fsm import run_fsm

APOGEE_TOL_FT = 15.0
TIMING_TOL_S = 0.5


def evaluate_run(seed, thrust_scale=1.0, drag_scale=1.0, early_burnout_s=None,
                  dropout_window=None, noise_multiplier=1.0):
    truth = simulate_flight(thrust_scale=thrust_scale, drag_scale=drag_scale,
                             early_burnout_s=early_burnout_s, seed=seed)
    sensors = synthesize_sensors(truth, dropout_window=dropout_window,
                                  noise_multiplier=noise_multiplier, seed=seed)

    crashed = False
    try:
        kf_out = run_filter(sensors)
        fsm_out = run_fsm(kf_out)
    except Exception as e:  # pragma: no cover - safety net for the harness itself
        crashed = True
        kf_out, fsm_out = None, None

    true_apogee_ft = truth["apogee_altitude_m"] * 3.28084

    result = {
        "seed": seed, "crashed": crashed,
        "true_apogee_ft": true_apogee_ft,
        "true_apogee_t": truth["apogee_t"],
    }

    if crashed:
        result.update({"kf_apogee_ft": None, "apogee_err_ft": None,
                        "deploy_t": None, "timing_err_s": None,
                        "state_sequence": None, "false_transition": None})
        return result

    kf_apogee_ft = float(np.max(kf_out["altitude"])) * 3.28084
    apogee_err_ft = kf_apogee_ft - true_apogee_ft
    deploy_t = fsm_out["deploy_t"]
    timing_err_s = (deploy_t - truth["apogee_t"]) if deploy_t is not None else None

    seq = [s for _, s in fsm_out["transitions"]]
    expected_prefix = ["BOOST", "COAST", "APOGEE", "DESCENT"]
    false_transition = seq[:4] != expected_prefix

    result.update({
        "kf_apogee_ft": kf_apogee_ft, "apogee_err_ft": apogee_err_ft,
        "deploy_t": deploy_t, "timing_err_s": timing_err_s,
        "state_sequence": seq, "false_transition": false_transition,
    })
    return result


def run_nominal_suite(n_runs=20):
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_runs):
        thrust_scale = 1.0 + rng.uniform(-0.05, 0.05)
        drag_scale = 1.0 + rng.uniform(-0.02, 0.02)
        r = evaluate_run(seed=100 + i, thrust_scale=thrust_scale, drag_scale=drag_scale)
        r["run_id"] = i + 1
        r["thrust_scale"] = thrust_scale
        r["drag_scale"] = drag_scale
        rows.append(r)
    return rows


def summarize(rows):
    n = len(rows)
    apogee_pass = sum(1 for r in rows if not r["crashed"] and abs(r["apogee_err_ft"]) <= APOGEE_TOL_FT)
    timing_pass = sum(1 for r in rows if not r["crashed"] and r["timing_err_s"] is not None
                       and abs(r["timing_err_s"]) <= TIMING_TOL_S)
    false_trans = sum(1 for r in rows if not r["crashed"] and r["false_transition"])
    crashes = sum(1 for r in rows if r["crashed"])
    return {
        "n_runs": n,
        "apogee_pass": apogee_pass, "apogee_pass_rate": apogee_pass / n,
        "timing_pass": timing_pass, "timing_pass_rate": timing_pass / n,
        "false_transitions": false_trans,
        "crashes": crashes,
    }


def write_csv(rows, path):
    fieldnames = ["run_id", "seed", "thrust_scale", "drag_scale", "true_apogee_ft",
                  "kf_apogee_ft", "apogee_err_ft", "true_apogee_t", "deploy_t",
                  "timing_err_s", "false_transition", "crashed"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    print("=" * 70)
    print("SIL VALIDATION -- 20-run nominal suite")
    print("=" * 70)
    rows = run_nominal_suite(20)
    for r in rows:
        status = "CRASH" if r["crashed"] else (
            "PASS" if abs(r["apogee_err_ft"]) <= APOGEE_TOL_FT and
            abs(r["timing_err_s"]) <= TIMING_TOL_S and not r["false_transition"]
            else "FAIL")
        print(f"  run {r['run_id']:2d}  thrust x{r['thrust_scale']:.3f}  drag x{r['drag_scale']:.3f}  "
              f"apogee_err={r['apogee_err_ft']:+.2f}ft  timing_err={r['timing_err_s']:+.3f}s  {status}")

    summary = summarize(rows)
    print("-" * 70)
    print(summary)
    write_csv(rows, os.path.join(os.path.dirname(__file__), "..", "docs", "sil_nominal_results.csv"))

    print("\n" + "=" * 70)
    print("STRESS TEST 1 -- Sensor dropout (2s baro dropout during COAST)")
    print("=" * 70)
    r = evaluate_run(seed=500, dropout_window=(3.0, 5.0))
    print(r)

    print("\n" + "=" * 70)
    print("STRESS TEST 2 -- Noise stress (2x nominal sensor noise)")
    print("=" * 70)
    r2 = evaluate_run(seed=501, noise_multiplier=2.0)
    print(r2)

    print("\n" + "=" * 70)
    print("STRESS TEST 3 -- Early burnout (motor cuts out 0.5s early)")
    print("=" * 70)
    r3 = evaluate_run(seed=502, early_burnout_s=1.92)
    print(r3)
