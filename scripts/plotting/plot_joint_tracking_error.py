
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "outputs/free_trot_v3/free_trot_v3_log.csv"
OUTPUT_PATH = ROOT / "outputs/free_trot_v3/joint_tracking_error.png"

# 读取实验数据
df = pd.read_csv(CSV_PATH)
time = df["time"]

# 四条腿及其关节
legs = ["FR", "FL", "RR", "RL"]
joints = ["hip", "thigh", "calf"]

# 颜色与关节对应
colors = {
    "hip": "#2563eb",
    "thigh": "#dc2626",
    "calf": "#16a34a"
}

# 创建 2×2 子图
fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 8),
    dpi=160,
    sharex=True,
    sharey=True
)

axes = axes.flatten()

for ax, leg in zip(axes, legs):

    # 实验阶段背景
    ax.axvspan(0, 2, color="#94a3b8", alpha=0.16)
    ax.axvspan(2, 3, color="#f59e0b", alpha=0.16)
    ax.axvspan(3, 8, color="#22c55e", alpha=0.10)

    # 绘制该腿三个关节的跟踪误差
    for joint in joints:
        des_col = f"{leg}_{joint}_joint_des"
        actual_col = f"{leg}_{joint}_joint_actual"

        error = df[actual_col] - df[des_col]

        ax.plot(
            time,
            error,
            color=colors[joint],
            linewidth=1.0,
            label=joint.capitalize()
        )

    # 阶段分界线和零误差参考线
    ax.axvline(2, color="#d97706", linestyle="--", linewidth=0.9)
    ax.axvline(3, color="#16a34a", linestyle="--", linewidth=0.9)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)

    ax.set_title(f"{leg} Leg", fontsize=12)
    ax.set_xlim(0, 8)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, ncol=3)

# 统一坐标轴
fig.supxlabel("Time (s)", fontsize=11)
fig.supylabel("Joint tracking error (rad)", fontsize=11)
fig.suptitle(
    "Go2 Joint Position Tracking Errors During Trot Locomotion",
    fontsize=14,
    y=0.99
)

fig.tight_layout(rect=[0.03, 0.03, 1, 0.96])

# 保存图片
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, bbox_inches="tight")
plt.close(fig)

# 输出各关节误差统计
all_errors = []

print("Joint tracking error statistics:")
for leg in legs:
    for joint in joints:
        des_col = f"{leg}_{joint}_joint_des"
        actual_col = f"{leg}_{joint}_joint_actual"

        error = df[actual_col] - df[des_col]
        rmse = (error.pow(2).mean()) ** 0.5
        max_error = error.abs().max()

        all_errors.append(error)

        print(
            f"{leg}_{joint}: "
            f"RMSE={rmse:.6f} rad, "
            f"Max |error|={max_error:.6f} rad"
        )

overall_error = pd.concat(all_errors, axis=1).to_numpy().flatten()
overall_rmse = (overall_error ** 2).mean() ** 0.5

print(f"\nOverall RMSE: {overall_rmse:.6f} rad")
print(f"Image saved to: {OUTPUT_PATH}")
