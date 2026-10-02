import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入正运动学函数
sys.path.insert(0, "scripts")
from leg_kinematics import forward_kinematics


# =========================
# 1. 文件路径
# =========================

PD_FILE = Path("outputs/fixed_pd/pd_tracking.csv")
REF_FILE = Path("outputs/trot/trot_trajectory.csv")

OUT_DIR = Path("outputs/fixed_pd/foot_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 读取数据
# =========================

pd_df = pd.read_csv(PD_FILE)
ref_df = pd.read_csv(REF_FILE)

legs = ["FR", "FL", "RR", "RL"]

t_actual = pd_df["time"].to_numpy()
t_ref = ref_df["time"].to_numpy()

# 仅分析规划轨迹和PD仿真时间重叠的部分
valid = (
    (t_actual >= t_ref[0])
    & (t_actual <= t_ref[-1])
)

t = t_actual[valid]

if len(t) == 0:
    raise ValueError("两组数据没有重叠的时间范围。")


# =========================
# 3. 计算实际足端位置
# =========================

results = {}

for leg in legs:

    joint_names = [
        f"{leg}_hip_joint_actual",
        f"{leg}_thigh_joint_actual",
        f"{leg}_calf_joint_actual",
    ]

    q_actual = pd_df.loc[valid, joint_names].to_numpy()

    # FK计算实际足端位置
    p_actual = np.array([
        forward_kinematics(leg, q)
        for q in q_actual
    ])

    # 读取规划足端位置
    ref_names = [
        f"{leg}_x",
        f"{leg}_y",
        f"{leg}_z",
    ]

    p_ref_original = ref_df[ref_names].to_numpy()

    # 将规划轨迹插值到实际仿真时间点
    p_ref = np.column_stack([
        np.interp(t, t_ref, p_ref_original[:, axis])
        for axis in range(3)
    ])

    # 足端位置误差向量
    error_vector = p_actual - p_ref

    # 欧氏距离误差
    error_norm = np.linalg.norm(error_vector, axis=1)

    results[leg] = {
        "actual": p_actual,
        "reference": p_ref,
        "error_vector": error_vector,
        "error_norm": error_norm,
    }


# =========================
# 4. 误差统计
# =========================

metrics = []

for leg in legs:

    e = results[leg]["error_norm"]

    metrics.append({
        "leg": leg,
        "RMSE_m": np.sqrt(np.mean(e ** 2)),
        "MAE_m": np.mean(np.abs(e)),
        "MaxError_m": np.max(e),
        "MeanError_m": np.mean(e),
    })

metrics_df = pd.DataFrame(metrics)

# 四条腿合并统计
all_errors = np.concatenate([
    results[leg]["error_norm"]
    for leg in legs
])

overall = pd.DataFrame([{
    "leg": "ALL",
    "RMSE_m": np.sqrt(np.mean(all_errors ** 2)),
    "MAE_m": np.mean(np.abs(all_errors)),
    "MaxError_m": np.max(all_errors),
    "MeanError_m": np.mean(all_errors),
}])

metrics_df = pd.concat(
    [metrics_df, overall],
    ignore_index=True
)

metrics_df.to_csv(
    OUT_DIR / "foot_metrics.csv",
    index=False
)

print("\n========== 足端跟踪误差统计 ==========")
print(metrics_df.to_string(index=False))


# =========================
# 5. 保存逐时刻误差数据
# =========================

output = {"time": t}

for leg in legs:

    data = results[leg]

    for axis, name in enumerate(["x", "y", "z"]):
        output[f"{leg}_{name}_actual"] = data["actual"][:, axis]
        output[f"{leg}_{name}_ref"] = data["reference"][:, axis]
        output[f"{leg}_{name}_error"] = data["error_vector"][:, axis]

    output[f"{leg}_error_norm"] = data["error_norm"]

tracking_df = pd.DataFrame(output)

tracking_df.to_csv(
    OUT_DIR / "foot_tracking.csv",
    index=False
)


# =========================
# 6. 绘制足端轨迹对比图
# =========================

fig, axes = plt.subplots(
    4, 3,
    figsize=(16, 12),
    sharex=True
)

axes = np.asarray(axes)

axis_names = ["x", "y", "z"]

for i, leg in enumerate(legs):

    data = results[leg]

    for j, axis in enumerate(axis_names):

        ax = axes[i, j]

        ax.plot(
            t,
            data["reference"][:, j],
            label="Reference",
            linewidth=1.5
        )

        ax.plot(
            t,
            data["actual"][:, j],
            label="Actual",
            linestyle="--",
            linewidth=1.2
        )

        ax.set_title(f"{leg} Foot - {axis.upper()}")
        ax.set_ylabel("Position (m)")
        ax.grid(True, alpha=0.3)

        if i == 0 and j == 0:
            ax.legend()

for ax in axes[-1, :]:
    ax.set_xlabel("Time (s)")

fig.suptitle(
    "Foot Position Tracking: Reference vs Actual",
    fontsize=15
)

fig.tight_layout()

fig.savefig(
    OUT_DIR / "foot_tracking.png",
    dpi=200
)

plt.close(fig)


# =========================
# 7. 绘制足端误差曲线
# =========================

fig, ax = plt.subplots(figsize=(12, 6))

for leg in legs:

    ax.plot(
        t,
        results[leg]["error_norm"],
        label=leg,
        linewidth=1.2
    )

ax.set_xlabel("Time (s)")
ax.set_ylabel("Foot Position Error (m)")
ax.set_title("Foot Position Tracking Error")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()

fig.savefig(
    OUT_DIR / "foot_error.png",
    dpi=200
)

plt.close(fig)


print("\n分析完成！")
print(f"结果保存至：{OUT_DIR}")
