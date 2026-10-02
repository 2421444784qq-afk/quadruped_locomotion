import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "outputs/free_trot_v3/free_trot_v3_log.csv"
OUTPUT_PATH = ROOT / "outputs/free_trot_v3/forward_velocity.png"

# 读取实验数据
df = pd.read_csv(CSV_PATH)

time = df["time"]
actual_velocity = df["base_vx_world"]

# 目标速度
target_velocity = 0.5

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)

# 实验阶段背景
ax.axvspan(0, 2, color="#94a3b8", alpha=0.16, label="Stand")
ax.axvspan(2, 3, color="#f59e0b", alpha=0.16, label="Transition")
ax.axvspan(3, 8, color="#22c55e", alpha=0.10, label="Trot")

# 实际速度
ax.plot(
    time,
    actual_velocity,
    color="#2563eb",
    linewidth=1.5,
    label="Actual velocity",
    zorder=3
)

# 目标速度，仅在 Trot 阶段显示
ax.plot(
    [3, 8],
    [target_velocity, target_velocity],
    color="#dc2626",
    linestyle="--",
    linewidth=1.8,
    label="Target velocity (0.5 m/s)"
)

# 阶段分界线
ax.axvline(2, color="#d97706", linestyle="--", linewidth=1)
ax.axvline(3, color="#16a34a", linestyle="--", linewidth=1)

# 坐标轴与标题
ax.set_title("Go2 Forward Velocity Tracking", fontsize=13)
ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Forward velocity (m/s)", fontsize=11)
ax.set_xlim(0, 8)

ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="upper right", frameon=True)

fig.tight_layout()

# 保存图片
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, bbox_inches="tight")
plt.close(fig)

# 输出统计信息
trot_df = df[df["time"] >= 3]
mean_velocity = trot_df["base_vx_world"].mean()

print(f"图像已保存至: {OUTPUT_PATH}")
print(f"Trot 阶段平均实际速度: {mean_velocity:.4f} m/s")
print(f"目标速度: {target_velocity:.4f} m/s")
print(f"平均速度误差: {target_velocity - mean_velocity:.4f} m/s")
