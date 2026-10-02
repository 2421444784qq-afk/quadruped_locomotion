
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# 1. 步态参数
# =========================

T = 2.0                  # 步态周期 (s)
STEP_LENGTH = 0.08       # 足端前后运动总距离 (m)
STEP_HEIGHT = 0.05       # 摆动相最大抬腿高度 (m)
DUTY_FACTOR = 0.6        # 支撑相占比
FREQUENCY = 100          # 采样频率 (Hz)

NUM_CYCLES = 2           # 绘制两个完整周期


# =========================
# 2. 四条腿默认足端位置
# 均为相对 base_link 的坐标
# =========================

FOOT_HOME = {
    "FR": np.array([0.1934, -0.142, -0.26480585]),
    "FL": np.array([0.1934,  0.142, -0.26480585]),
    "RR": np.array([-0.1934, -0.142, -0.26480585]),
    "RL": np.array([-0.1934,  0.142, -0.26480585]),
}


# 对角腿同步
PHASE_OFFSETS = {
    "FR": 0.0,
    "RL": 0.0,
    "FL": 0.5,
    "RR": 0.5,
}


# =========================
# 3. 单腿足端轨迹
# =========================

def generate_leg_trajectory(t, leg_name):
    """
    生成单条腿相对机身的足端轨迹。

    支撑相：
        足端从前方向后方移动，保持高度不变。

    摆动相：
        足端从后方向前方摆动，并抬高。
    """

    center = FOOT_HOME[leg_name]

    phase = (
        t / T + PHASE_OFFSETS[leg_name]
    ) % 1.0

    x = np.zeros_like(t)
    z = np.zeros_like(t)

    for i, p in enumerate(phase):

        if p < DUTY_FACTOR:
            # 支撑相：从前向后
            s = p / DUTY_FACTOR

            x[i] = STEP_LENGTH / 2 - STEP_LENGTH * s
            z[i] = 0.0

        else:
            # 摆动相：从后向前
            s = (p - DUTY_FACTOR) / (1.0 - DUTY_FACTOR)

            x[i] = (
                -STEP_LENGTH / 2
                + STEP_LENGTH * s
            )

            # 半余弦抬腿轨迹
            z[i] = (
                STEP_HEIGHT
                * 0.5
                * (1 - np.cos(2 * np.pi * s))
            )

    # 将局部轨迹叠加到各腿默认足端位置
    trajectory = np.zeros((len(t), 3))

    trajectory[:, 0] = center[0] + x
    trajectory[:, 1] = center[1]
    trajectory[:, 2] = center[2] + z

    return trajectory, phase


# =========================
# 4. 生成四腿轨迹
# =========================

dt = 1.0 / FREQUENCY
t = np.arange(
    0,
    NUM_CYCLES * T,
    dt
)

trajectories = {}
phases = {}

for leg in FOOT_HOME:
    trajectories[leg], phases[leg] = (
        generate_leg_trajectory(t, leg)
    )


# =========================
# 5. 输出轨迹范围
# =========================

print("========== Trot 步态轨迹 ==========")
print(f"步态周期: {T:.2f} s")
print(f"步长: {STEP_LENGTH:.3f} m")
print(f"抬腿高度: {STEP_HEIGHT:.3f} m")
print(f"支撑相占比: {DUTY_FACTOR:.2f}")
print(f"采样频率: {FREQUENCY} Hz")

for leg, traj in trajectories.items():
    print(f"\n{leg}:")
    print(
        f"  X范围: [{traj[:, 0].min():.4f}, "
        f"{traj[:, 0].max():.4f}] m"
    )
    print(
        f"  Y范围: [{traj[:, 1].min():.4f}, "
        f"{traj[:, 1].max():.4f}] m"
    )
    print(
        f"  Z范围: [{traj[:, 2].min():.4f}, "
        f"{traj[:, 2].max():.4f}] m"
    )


# =========================
# 6. 绘制足端轨迹
# =========================

fig = plt.figure(figsize=(15, 10))

# X-Z 轨迹
ax1 = fig.add_subplot(221)

for leg, traj in trajectories.items():
    ax1.plot(
        traj[:, 0],
        traj[:, 2],
        label=leg
    )

ax1.set_title("Foot Trajectory (X-Z)")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Z (m)")
ax1.grid(True)
ax1.legend()
ax1.axis("equal")


# X-Y 轨迹
ax2 = fig.add_subplot(222)

for leg, traj in trajectories.items():
    ax2.plot(
        traj[:, 0],
        traj[:, 1],
        label=leg
    )

ax2.set_title("Foot Trajectory (X-Y)")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.grid(True)
ax2.legend()
ax2.axis("equal")


# 足端高度随时间变化
ax3 = fig.add_subplot(223)

for leg, traj in trajectories.items():
    ax3.plot(
        t,
        traj[:, 2],
        label=leg
    )

ax3.set_title("Foot Height vs Time")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Z (m)")
ax3.grid(True)
ax3.legend()


# 步态相位
ax4 = fig.add_subplot(224)

for leg, phase in phases.items():
    ax4.plot(
        t,
        phase,
        label=leg
    )

ax4.set_title("Gait Phase")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Normalized Phase")
ax4.grid(True)
ax4.legend()

plt.tight_layout()


# =========================
# 7. 保存数据和图像
# =========================

output_dir = Path("outputs/trot")
output_dir.mkdir(parents=True, exist_ok=True)

csv_data = {"time": t}

for leg, traj in trajectories.items():
    csv_data[f"{leg}_x"] = traj[:, 0]
    csv_data[f"{leg}_y"] = traj[:, 1]
    csv_data[f"{leg}_z"] = traj[:, 2]
    csv_data[f"{leg}_phase"] = phases[leg]

import pandas as pd

pd.DataFrame(csv_data).to_csv(
    output_dir / "trot_trajectory.csv",
    index=False
)

fig.savefig(
    output_dir / "trot_trajectory.png",
    dpi=200,
    bbox_inches="tight"
)

print("\n数据已保存至:", output_dir)
print("  trot_trajectory.csv")
print("  trot_trajectory.png")

plt.show()
