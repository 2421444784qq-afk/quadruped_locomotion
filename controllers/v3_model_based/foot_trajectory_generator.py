import numpy as np

from controllers.v3_model_based.gait_scheduler import GaitScheduler
from controllers.v3_model_based.swing_leg_controller import SwingLegController
from controllers.v3_model_based.foothold_planner import FootholdPlanner


class FootTrajectoryGenerator:
    """
    四足周期性足端轨迹生成器。

    支撑相：
        足端相对机身从前向后移动。

    摆动相：
        足端从后向前回摆，并完成抬腿。

    支持两种模式：
        feedback_enabled=False：固定落脚点（对照组）
        feedback_enabled=True ：基于机身速度的动态落脚点

    动态落脚偏移在摆动相开始时更新，并在该摆动周期内锁定。
    """

    LEGS = ("FR", "FL", "RR", "RL")

    def __init__(
        self,
        period=0.5,
        duty_factor=0.6,
        step_length=0.08,
        clearance=0.05,
        foot_centers=None,
        feedback_gain=0.15,
        max_foothold_offset=0.05,
    ):
        self.gait = GaitScheduler(period, duty_factor)
        self.swing = SwingLegController(clearance)
        self.foothold_planner = FootholdPlanner(
            gain=feedback_gain,
            max_offset=max_foothold_offset,
        )

        self.period = period
        self.duty_factor = duty_factor
        self.step_length = step_length

        if foot_centers is None:
            foot_centers = {
                "FR": np.array([0.19, -0.12, -0.25]),
                "FL": np.array([0.19,  0.12, -0.25]),
                "RR": np.array([-0.19, -0.12, -0.25]),
                "RL": np.array([-0.19,  0.12, -0.25]),
            }

        self.foot_centers = {
            leg: np.asarray(foot_centers[leg], dtype=float).copy()
            for leg in self.LEGS
        }

        self.front_positions = {}
        self.rear_positions = {}

        for leg in self.LEGS:
            center = self.foot_centers[leg]

            front = center.copy()
            rear = center.copy()

            front[0] += step_length / 2
            rear[0] -= step_length / 2

            self.front_positions[leg] = front
            self.rear_positions[leg] = rear

        # 每条腿当前支撑轨迹所使用的偏移。
        self.current_offsets = {
            leg: 0.0 for leg in self.LEGS
        }

        # 每条腿当前摆动周期锁定的落脚偏移。
        self.locked_offsets = {
            leg: 0.0 for leg in self.LEGS
        }

        self.last_modes = {
            leg: None for leg in self.LEGS
        }

        self.initialized = {
            leg: False for leg in self.LEGS
        }

    def reset(self):
        """重置落脚偏移与相位切换状态。"""
        for leg in self.LEGS:
            self.current_offsets[leg] = 0.0
            self.locked_offsets[leg] = 0.0
            self.last_modes[leg] = None
            self.initialized[leg] = False

    def get_targets(
        self,
        time,
        body_velocity_x=0.0,
        desired_velocity_x=0.0,
        feedback_enabled=False,
    ):
        """
        生成四条腿的足端目标。

        参数：
            time:
                当前步态时间，单位 s。

            body_velocity_x:
                当前机身前向速度，单位 m/s。

            desired_velocity_x:
                期望机身前向速度，单位 m/s。

            feedback_enabled:
                是否启用动态落脚点反馈。
        """
        gait_state = self.gait.get_state(time)
        targets = {}

        stance_duration = self.period * self.duty_factor
        swing_duration = self.period * (1.0 - self.duty_factor)

        for leg in self.LEGS:
            state = gait_state[leg]
            mode = state["mode"]
            phase = state["normalized_phase"]

            # 初始化：如果首次调用时处于摆动相，也能正确生成目标。
            if not self.initialized[leg]:
                if mode == "swing" and feedback_enabled:
                    nominal = self.front_positions[leg]
                    _, offset = self.foothold_planner.compute(
                        nominal,
                        body_velocity_x,
                        desired_velocity_x,
                    )
                    self.current_offsets[leg] = offset
                    self.locked_offsets[leg] = offset

                self.initialized[leg] = True

            # 仅在支撑相切换到摆动相时更新一次落脚偏移。
            if (
                mode == "swing"
                and self.last_modes[leg] == "stance"
                and feedback_enabled
            ):
                nominal = self.front_positions[leg]

                _, offset = self.foothold_planner.compute(
                    nominal,
                    body_velocity_x,
                    desired_velocity_x,
                )

                self.locked_offsets[leg] = offset

            # 固定模式下偏移始终为零。
            if not feedback_enabled:
                self.current_offsets[leg] = 0.0
                self.locked_offsets[leg] = 0.0

            if mode == "stance":
                # 支撑相沿用当前偏移，从前方向后方移动。
                offset = self.current_offsets[leg]

                start = self.front_positions[leg].copy()
                end = self.rear_positions[leg].copy()

                start[0] += offset
                end[0] += offset

                position = start + phase * (end - start)
                velocity = (end - start) / stance_duration

            else:
                # 摆动起点沿用当前支撑偏移；
                # 摆动终点使用本次锁定的落脚偏移。
                start = self.rear_positions[leg].copy()
                end = self.front_positions[leg].copy()

                start[0] += self.current_offsets[leg]
                end[0] += self.locked_offsets[leg]

                position, velocity = self.swing.compute(
                    start,
                    end,
                    phase,
                    swing_duration,
                )

                # 摆动结束后，新的偏移成为下一支撑相的偏移。
                # 此处只更新内部状态，不改变本次摆动轨迹目标。
                self.current_offsets[leg] = self.locked_offsets[leg]

            targets[leg] = {
                "position": position,
                "velocity": velocity,
                "in_contact": state["in_contact"],
                "mode": mode,
                "phase": state["phase"],
                "normalized_phase": phase,
                "foothold_offset": self.locked_offsets[leg],
            }

            self.last_modes[leg] = mode

        return targets
