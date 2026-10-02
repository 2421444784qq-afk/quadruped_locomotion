import numpy as np


class GaitScheduler:
    """
    对角小跑（Trot）步态调度器。

    FR + RL：同相
    FL + RR：相差半个周期

    phase < duty_factor：支撑相
    phase >= duty_factor：摆动相
    """

    LEGS = ("FR", "FL", "RR", "RL")

    def __init__(self, period=0.5, duty_factor=0.6):
        if period <= 0:
            raise ValueError("period must be positive")
        if not 0 < duty_factor < 1:
            raise ValueError("duty_factor must be between 0 and 1")

        self.period = period
        self.duty_factor = duty_factor

        self.phase_offsets = {
            "FR": 0.0,
            "FL": 0.5,
            "RR": 0.5,
            "RL": 0.0,
        }

    def get_state(self, time):
        result = {}

        for leg in self.LEGS:
            phase = (
                time / self.period + self.phase_offsets[leg]
            ) % 1.0

            in_contact = phase < self.duty_factor

            if in_contact:
                normalized_phase = phase / self.duty_factor
                mode = "stance"
            else:
                normalized_phase = (
                    phase - self.duty_factor
                ) / (1.0 - self.duty_factor)
                mode = "swing"

            result[leg] = {
                "phase": phase,
                "normalized_phase": normalized_phase,
                "in_contact": in_contact,
                "mode": mode,
            }

        return result
