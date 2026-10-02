import numpy as np


class FootPlacementController:
    """
    基于机身前向速度的逐腿落脚点反馈控制器。

    控制特点：
    1. 每条腿独立维护反馈偏移量。
    2. 仅在摆动相开始时更新目标偏移量。
    3. 摆动相内平滑过渡到新偏移量。
    4. 支撑相内保持偏移量不变。
    5. 只修改足端 X 坐标，不改变 Y、Z。
    """

    def __init__(
        self,
        leg_names=("FR", "FL", "RR", "RL"),
        desired_velocity=0.10,
        velocity_gain=0.10,
        max_offset=0.04,
        duty_factor=0.6
    ):
        self.leg_names = tuple(leg_names)
        self.desired_velocity = float(desired_velocity)
        self.velocity_gain = float(velocity_gain)
        self.max_offset = float(max_offset)
        self.duty_factor = float(duty_factor)

        if not 0.0 < self.duty_factor < 1.0:
            raise ValueError(
                "duty_factor must be between 0 and 1"
            )

        if self.max_offset < 0.0:
            raise ValueError(
                "max_offset must be non-negative"
            )

        self.current_offsets = {
            leg: 0.0 for leg in self.leg_names
        }

        self.target_offsets = {
            leg: 0.0 for leg in self.leg_names
        }

        self.previous_phases = {
            leg: None for leg in self.leg_names
        }

        self.swing_start_offsets = {
            leg: 0.0 for leg in self.leg_names
        }

    def compute_offset(self, v_forward):
        """
        根据机身前向速度计算目标偏移量。
        """
        velocity_error = (
            float(v_forward) - self.desired_velocity
        )

        offset = (
            self.velocity_gain * velocity_error
        )

        return float(
            np.clip(
                offset,
                -self.max_offset,
                self.max_offset
            )
        )

    def _is_swing(self, phase):
        """
        判断当前是否处于摆动相。
        phase 使用 [0, 1) 范围内的周期相位。
        """
        return phase >= self.duty_factor

    def _smoothstep(self, u):
        """
        三次平滑插值，输入和输出范围均为 [0, 1]。
        """
        u = float(np.clip(u, 0.0, 1.0))
        return u * u * (3.0 - 2.0 * u)

    def update(
        self,
        leg,
        nominal_position,
        phase,
        v_forward
    ):
        """
        更新指定腿的足端目标。

        参数：
            leg: 腿名
            nominal_position: 名义足端位置，base_link 坐标系
            phase: 当前腿的周期相位，[0, 1)
            v_forward: 机身坐标系下的前向速度

        返回：
            修正后的足端目标位置
        """
        if leg not in self.current_offsets:
            raise ValueError(f"Unknown leg: {leg}")

        position = np.asarray(
            nominal_position,
            dtype=float
        ).copy()

        if position.shape != (3,):
            raise ValueError(
                "nominal_position must have shape (3,)"
            )

        phase = float(phase % 1.0)
        in_swing = self._is_swing(phase)

        previous_phase = self.previous_phases[leg]

        # 检测支撑相 -> 摆动相的切换
        entering_swing = (
            in_swing
            and (
                previous_phase is None
                or previous_phase < self.duty_factor
            )
        )

        if entering_swing:
            # 保存当前偏移，作为本次平滑过渡的起点
            self.swing_start_offsets[leg] = (
                self.current_offsets[leg]
            )

            # 根据当前机身速度计算本次摆动的目标偏移
            self.target_offsets[leg] = (
                self.compute_offset(v_forward)
            )

        if in_swing:
            # 摆动相归一化进度
            u = (
                phase - self.duty_factor
            ) / (
                1.0 - self.duty_factor
            )

            blend = self._smoothstep(u)

            start = self.swing_start_offsets[leg]
            target = self.target_offsets[leg]

            self.current_offsets[leg] = (
                start + (target - start) * blend
            )

        # 支撑相不更新 current_offsets，保持上次落脚偏移
        self.previous_phases[leg] = phase

        position[0] += self.current_offsets[leg]

        return position