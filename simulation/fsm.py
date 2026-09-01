"""
fsm.py
Finite state machine for autonomous flight phase detection, driven by the
Kalman filter's altitude/velocity/acceleration estimate. Implements
N-sample hysteresis on every transition to reject noise-induced false
triggers, and predictive apogee detection (t_apogee = v / g < 0.2 s)
rather than simple sign-of-velocity detection.
"""

import numpy as np

STATES = ["IDLE", "BOOST", "COAST", "APOGEE", "DESCENT", "LANDED"]
G = 9.81
HYSTERESIS_SAMPLES = 5       # samples a condition must hold before transition
APOGEE_PREDICT_THRESHOLD_S = 0.2
LANDED_SPEED_THRESHOLD_MPS = 1.0
LANDED_HOLD_S = 3.0


class FlightStateMachine:
    def __init__(self, dt):
        self.dt = dt
        self.state = "IDLE"
        self._counter = 0
        self.deploy_flag = False
        self.landed_hold_samples = 0
        self.log = []  # (t, state)

    def _try_transition(self, condition, next_state):
        if condition:
            self._counter += 1
        else:
            self._counter = 0
        if self._counter >= HYSTERESIS_SAMPLES:
            self.state = next_state
            self._counter = 0
            return True
        return False

    def step(self, t, alt, vel, accel):
        prev_state = self.state

        if self.state == "IDLE":
            self._try_transition(accel > 2 * G, "BOOST")

        elif self.state == "BOOST":
            self._try_transition(accel < 0, "COAST")

        elif self.state == "COAST":
            t_apogee = vel / G if vel > 0 else 0.0
            if self._try_transition(0 <= t_apogee < APOGEE_PREDICT_THRESHOLD_S, "APOGEE"):
                self.deploy_flag = True

        elif self.state == "APOGEE":
            # single-sample dwell then move to DESCENT; deploy_flag stays latched
            self.state = "DESCENT"

        elif self.state == "DESCENT":
            if vel < 0 and abs(vel) < LANDED_SPEED_THRESHOLD_MPS:
                self.landed_hold_samples += 1
            else:
                self.landed_hold_samples = 0
            if self.landed_hold_samples >= int(LANDED_HOLD_S / self.dt):
                self.state = "LANDED"

        if self.state != prev_state:
            self.log.append((t, self.state))

        return self.state


def run_fsm(kf_output):
    t, alt, vel, acc = (kf_output["t"], kf_output["altitude"],
                         kf_output["velocity"], kf_output["acceleration"])
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.01
    fsm = FlightStateMachine(dt)
    states = []
    deploy_t = None
    for i in range(len(t)):
        s = fsm.step(t[i], alt[i], vel[i], acc[i])
        states.append(s)
        if fsm.deploy_flag and deploy_t is None:
            deploy_t = t[i]
    return {
        "t": t, "state": states, "transitions": fsm.log,
        "deploy_t": deploy_t,
    }
