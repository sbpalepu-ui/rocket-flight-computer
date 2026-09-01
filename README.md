# Guided Rocket Flight Computer — Simulation & SIL Validation

Software core (simulation, Kalman filter, flight-phase FSM, and Monte Carlo
software-in-the-loop validation) for an autonomous rocket flight computer,
covering Phases 0–2 of the original project plan. No physical hardware was
built; see [`docs/Guided_Rocket_Flight_Computer_Report.pdf`](docs/Guided_Rocket_Flight_Computer_Report.pdf) (viewable on GitHub; `.docx` also included) for full scope,
results, and a debugging log of real issues found while building this.

## Structure
- `simulation/` — flight dynamics model, sensor synthesis, Kalman filter, FSM
- `sil/` — 20-run Monte Carlo + stress-test validation harness, plot generator
- `docs/` — technical report (PDF + .docx), figures, SIL results (.csv)

## Run it
```
cd simulation && python3 run_pipeline.py     # single reference flight + plot
cd sil && python3 sil_test.py                # full 20-run + stress-test suite
cd sil && python3 generate_plots.py          # regenerate all report figures
```

Requires: numpy, scipy, matplotlib (`pip install numpy scipy matplotlib`)
