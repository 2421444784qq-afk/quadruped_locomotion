import numpy as np
import matplotlib.pyplot as plt

from kinematics import forward_kinematics


class FootTrajectory:
    def __init__(
        self,
        center,
        step_length=0.08,
        step_height=0.05,
        duty_factor=0.6,
        period=2.0
    ):
        self.center = np.asarray(center, dtype=float)
        self.step_length = step_length
        self.step_height = step_height
        self.duty_factor = duty_factor
        self.period = period

    def get_position(self, t):
        """
        根据时间 t 返回足端目标位置。

        输出坐标系：base_link
        """
        phase = (t % self.period) / self.period

        x_front = self.center[0] + self.step_length / 2
        x_back = self.center[0] - self.step_length / 2

        # 支撑相：足端相对机身向后运动
        if phase < self.duty_factor:
            u = phase / self.duty_factor

            x = x_front + (x_back - x_front) * u
            z = self.center[2]

        # 摆动相：足端抬起并向前摆回
        else:
            u = (phase - self.duty_factor) / (1 - self.duty_factor)

            x = x_back + (x_front - x_back) * u
            z = (
                self.center[2]
                + self.step_height * np.sin(np.pi * u)
            )

        return np.array([x, self.center[1], z])


if __name__ == "__main__":
    # 当前已验证的站立关节角
    q_stand = np.array([0.0, 0.8, -1.4])

    # 通过正运动学得到基准足端位置
    p_stand = forward_kinematics(q_stand)

    trajectory = FootTrajectory(
        center=p_stand,
        step_length=0.08,
        step_height=0.05,
        duty_factor=0.6,
        period=2.0
    )

    # 生成一个周期的数据
    times = np.linspace(
        0,
        trajectory.period,
        500,
        endpoint=False
    )

    positions = np.array([
        trajectory.get_position(t)
        for t in times
    ])

    # 保存轨迹数据
    np.savetxt(
        "foot_trajectory.csv",
        np.column_stack((times, positions)),
        delimiter=",",
        header="time,x,y,z",
        comments=""
    )

    print("基准足端位置:", p_stand)
    print("轨迹数据已保存至 foot_trajectory.csv")
    print("X 范围:", positions[:, 0].min(), positions[:, 0].max())
    print("Z 范围:", positions[:, 2].min(), positions[:, 2].max())

    # 绘制 X-Z 平面轨迹
    plt.figure(figsize=(8, 5))
    plt.plot(
        positions[:, 0],
        positions[:, 2],
        label="Foot trajectory"
    )
    plt.scatter(
        [p_stand[0]],
        [p_stand[2]],
        color="red",
        label="Standing position"
    )

    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title("Go2 FR Foot Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("foot_trajectory.png", dpi=200)
    plt.show()
