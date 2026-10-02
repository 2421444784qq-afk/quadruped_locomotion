import numpy as np


class FootholdPlanner:
    """
    基于机身前向速度误差的动态落脚点规划器。

    坐标约定：
        x 正方向：机器人前方
        x 负方向：机器人后方

    输入：
        nominal_foothold: 名义落脚点 [x, y, z]
        body_velocity_x: 当前机身前向速度 (m/s)
        desired_velocity_x: 期望机身前向速度 (m/s)

    输出：
        foothold: 修正后的落脚点 [x, y, z]
        offset_x: 前后方向的落脚点偏移
    """

    def __init__(self, gain=0.15, max_offset=0.05):
        if gain < 0:
            raise ValueError("gain must be non-negative")
        if max_offset < 0:
            raise ValueError("max_offset must be non-negative")

        self.gain = gain
        self.max_offset = max_offset

    def compute(
        self,
        nominal_foothold,
        body_velocity_x,
        desired_velocity_x,
    ):
        nominal = np.asarray(nominal_foothold, dtype=float)

        if nominal.shape != (3,):
            raise ValueError("nominal_foothold must have shape (3,)")

        # 太快(body > desired)时，落脚点应往前放以刹车，故误差为 body - desired
        velocity_error = body_velocity_x - desired_velocity_x

        raw_offset = self.gain * velocity_error

        offset_x = float(
            np.clip(
                raw_offset,
                -self.max_offset,
                self.max_offset,
            )
        )

        foothold = nominal.copy()
        foothold[0] += offset_x

        return foothold, offset_x
