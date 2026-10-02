
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import sys

# 加入 scripts 目录，以便导入运动学模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from leg_kinematics import inverse_kinematics


# =========================
# 1. 文件路径
# =========================

INPUT_CSV = Path("outputs/trot/trot_trajectory.csv")
OUTPUT_DIR = Path("outputs/trot")


# =========================
# 2. 四腿关节配置
# =========================

LEGS = {
    "FR": {
        "joints": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
        ],
        "limits": [
            [-1.0472, 1.0472],
            [-1.5708, 3.4907],
            [-2.7227, -0.83776],
        ],
    },
    "FL": {
        "joints": [
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
        ],
        "limits": [
            [-1.0472, 1.0472],
            [-1.5708, 3.4907],
            [-2.7227, -0.83776],
        ],
    },
    "RR": {
        "joints": [
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ],
        "limits": [
            [-1.0472, 1.0472],
            [-0.5236, 4.5379],
            [-2.7227, -0.83776],
        ],
    },
    "RL": {
        "joints": [
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "limits": [
            [-1.0472, 1.0472],
            [-0.5236, 4.5379],
            [-2.7227, -0.83776],
        ],
    },
}


# 默认站立姿态
Q_HOME = np.array([0.0, 0.9, -1.8])


# =========================
# 3. 读取足端轨迹
# =========================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"找不到输入文件: {INPUT_CSV}"
    )

df = pd.read_csv(INPUT_CSV)
time = df["time"].to_numpy()

num_samples = len(time)

print("========== Trot 逆运动学求解 ==========")
print(f"轨迹点数量: {num_samples}")
print(f"轨迹时长: {time[-1] - time[0]:.2f} s")


# =========================
# 4. 逐时刻求解 IK
# =========================

joint_data = {}
ik_errors = {}
ik_success = {}
limit_violations = {}

for leg, cfg in LEGS.items():

    print(f"\n正在求解 {leg} ...")

    q_init = Q_HOME.copy()

    q_history = np.zeros((num_samples, 3))
    errors = np.zeros(num_samples)
    successes = np.zeros(num_samples, dtype=bool)
    violations = np.zeros(num_samples, dtype=bool)

    limits = np.array(cfg["limits"])

    for i in range(num_samples):

        target = np.array([
            df.loc[i, f"{leg}_x"],
            df.loc[i, f"{leg}_y"],
            df.loc[i, f"{leg}_z"],
        ])

        q, error, success = inverse_kinematics(
            leg=leg,
            target=target,
            q_init=q_init,
            damping=0.05,
            max_iter=100,
            tol=1e-5,
            max_step=0.2,
        )

        q_history[i] = q
        errors[i] = error
        successes[i] = success

        # 检查是否超出关节限位
        out_of_range = np.any(
            (q < limits[:, 0]) |
            (q > limits[:, 1])
        )

        violations[i] = out_of_range

        # 使用当前解作为下一时刻的初值
        q_init = q.copy()

    joint_data[leg] = q_history
    ik_errors[leg] = errors
    ik_success[leg] = successes
    limit_violations[leg] = violations

    print(
        f"  IK 收敛率: "
        f"{np.mean(successes) * 100:.2f}%"
    )
    print(
        f"  最大位置误差: "
        f"{np.max(errors):.8e} m"
    )
    print(
        f"  关节限位违规点: "
        f"{np.sum(violations)}"
    )


# =========================
# 5. 汇总结果
# =========================

print("\n========== 求解结果汇总 ==========")

all_success = True
all_within_limits = True

for leg in LEGS:

    success_rate = np.mean(ik_success[leg])
    violation_count = np.sum(limit_violations[leg])
    max_error = np.max(ik_errors[leg])

    if success_rate < 1.0:
        all_success = False

    if violation_count > 0:
        all_within_limits = False

    print(
        f"{leg}: "
        f"收敛率={success_rate * 100:.2f}%, "
        f"最大误差={max_error:.3e} m, "
        f"限位违规={violation_count}"
    )

print("\n整体 IK 收敛:", all_success)
print("全部关节满足限位:", all_within_limits)


# =========================
# 6. 保存关节角数据
# =========================

output_data = {"time": time}

for leg, cfg in LEGS.items():

    q_history = joint_data[leg]

    for j, joint_name in enumerate(cfg["joints"]):
        output_data[joint_name] = q_history[:, j]

    output_data[f"{leg}_ik_error"] = ik_errors[leg]
    output_data[f"{leg}_ik_success"] = ik_success[leg]
    output_data[f"{leg}_limit_violation"] = (
        limit_violations[leg]
    )

output_df = pd.DataFrame(output_data)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_path = OUTPUT_DIR / "trot_joint_trajectory.csv"

output_df.to_csv(
    output_path,
    index=False
)

print(f"\n关节角数据已保存: {output_path}")


# =========================
# 7. 绘制12个关节角轨迹
# =========================

fig, axes = plt.subplots(
    4, 3,
    figsize=(15, 12),
    sharex=True
)

for row, (leg, cfg) in enumerate(LEGS.items()):

    q_history = joint_data[leg]

    for j, joint_name in enumerate(cfg["joints"]):

        ax = axes[row, j]

        ax.plot(
            time,
            q_history[:, j],
            linewidth=1.2,
        )

        lower, upper = cfg["limits"][j]

        ax.axhline(
            lower,
            linestyle="--",
            linewidth=0.8,
            color="red",
            label="Lower limit",
        )

        ax.axhline(
            upper,
            linestyle="--",
            linewidth=0.8,
            color="red",
            label="Upper limit",
        )

        ax.set_title(joint_name)
        ax.set_ylabel("Angle (rad)")
        ax.grid(True, alpha=0.3)

        if row == 3:
            ax.set_xlabel("Time (s)")

fig.suptitle(
    "Trot Joint Reference Trajectories",
    fontsize=15,
)

plt.tight_layout()

figure_path = OUTPUT_DIR / "trot_joint_trajectory.png"

fig.savefig(
    figure_path,
    dpi=200,
    bbox_inches="tight",
)

print(f"关节角曲线已保存: {figure_path}")

plt.show()
