import numpy as np


class SwingLegController:
    """
    摆动腿足端轨迹生成器。

    水平轨迹：五次多项式
    竖直轨迹：64*s^3*(1-s)^3

    起点和终点的速度、加速度均为零。
    """

    def __init__(self, clearance=0.05):
        if clearance < 0:
            raise ValueError("clearance must be non-negative")
        self.clearance = clearance

    @staticmethod
    def _smoothstep5(s):
        return 10 * s**3 - 15 * s**4 + 6 * s**5

    @staticmethod
    def _smoothstep5_derivative(s):
        return 30 * s**2 - 60 * s**3 + 30 * s**4

    @staticmethod
    def _vertical_bump(s):
        return 64 * s**3 * (1 - s)**3

    @staticmethod
    def _vertical_bump_derivative(s):
        return 192 * s**2 * (1 - s)**2 * (1 - 2 * s)

    def compute(self, start_pos, end_pos, swing_phase, duration):
        start = np.asarray(start_pos, dtype=float)
        end = np.asarray(end_pos, dtype=float)

        if start.shape != (3,) or end.shape != (3,):
            raise ValueError("start_pos and end_pos must have shape (3,)")
        if duration <= 0:
            raise ValueError("duration must be positive")
        if not 0.0 <= swing_phase <= 1.0:
            raise ValueError("swing_phase must be in [0, 1]")

        s = swing_phase

        blend = self._smoothstep5(s)
        dblend = self._smoothstep5_derivative(s) / duration

        position = start + blend * (end - start)
        velocity = dblend * (end - start)

        bump = self._vertical_bump(s)
        dbump = self._vertical_bump_derivative(s) / duration

        position[2] += self.clearance * bump
        velocity[2] += self.clearance * dbump

        return position, velocity
