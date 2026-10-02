import numpy as np
import matplotlib.pyplot as plt

from foot_trajectory import FootTrajectory


class TrotGait:
    """
    四足机器人对角小跑步态生成器。

    腿的顺序：
    FR, FL, RR, RL

    FR 与 RL 同相；
    FL 与 RR 相差半个周期。
    """

    def __init__(
        self,
        centers,
        step_length=0.08,
        step_height=0.05,
        duty_factor=0.6,
        period=2.0
    ):
        self.leg_names = ["FR", "FL", "RR", "RL"]

        self.phase_offsets = {
            "FR": 0.0,
            "FL": 0.5,
            "RR": 0.5,
            "RL": 0.0
        }

        self.period = period

        self.trajectories = {}

        for name in self.leg_names:
            self.trajectories[name] = FootTrajectory(
                center=centers[name],
                step_length=step_length,
                step_height=step_height,
                duty_factor=duty_factor,
                period=period
            )

    def get_leg_position(self, leg_name, t):
        """
        获取指定腿在时刻 t 的足端目标位置。
        """
        phase_offset = self.phase_offsets[leg_name]

        phase_time = (
            t + phase_offset * self.period
        )

        return self.trajectories[
            leg_name
        ].get_position(phase_time)

    def get_all_positions(self, t):
        """
        返回四条腿的足端目标位置。
        """
        return {
            name: self.get_leg_position(name, t)
            for name in self.leg_names
        }


if __name__ == "__main__":

    # 使用统一的测试中心验证相位关系。
    # 后续接入机器人时，再替换为各条腿真实的站立足端位置。
    test_center = np.array([0.16, 0.0, -0.32])

    centers = {
        "FR": test_center.copy(),
        "FL": test_center.copy(),
        "RR": test_center.copy(),
        "RL": test_center.copy()
    }

    gait = TrotGait(
        centers=centers,
        step_length=0.08,
        step_height=0.05,
        duty_factor=0.6,
        period=2.0
    )

    times = np.linspace(
        0,
        gait.period,
        500,
        endpoint=False
    )

    # 记录四条腿的 X-Z 轨迹
    positions = {}

    for name in gait.leg_names:
        positions[name] = np.array([
            gait.get_leg_position(name, t)
            for t in times
        ])

    # 绘制四腿轨迹
    fig, axes = plt.subplots(
        2, 1,
        figsize=(10, 9)
    )

    for name in gait.leg_names:
        p = positions[name]

        axes[0].plot(
            p[:, 0],
            p[:, 2],
            label=name
        )

        axes[1].plot(
            times,
            p[:, 2],
            label=name
        )

    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Z (m)")
    axes[0].set_title("Trot Foot Trajectories")
    axes[0].axis("equal")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Foot Height Z (m)")
    axes[1].set_title("Foot Height vs Time")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("trot_gait_validation.png", dpi=200)
    plt.show()

    # 验证对角腿相位
    print("\n========== Trot 相位验证 ==========")

    for t in [0.0, 0.5, 1.0, 1.5]:
        result = gait.get_all_positions(t)

        print(f"\nt = {t:.1f} s")

        for name, pos in result.items():
            print(
                f"{name}: "
                f"x={pos[0]:.4f}, "
                f"z={pos[2]:.4f}"
            )

    print("\n验证图已保存：trot_gait_validation.png")
