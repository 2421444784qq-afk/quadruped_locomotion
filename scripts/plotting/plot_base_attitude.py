import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "outputs/free_trot_v3/free_trot_v3_log.csv"
OUTPUT_PATH = ROOT / "outputs/free_trot_v3/base_attitude.png"

# 读取实验数据
df = pd.read_csv(CSV_PATH)

time = df["time"]
roll_deg = df["roll"] * 180 / 3.141592653589793
pitch_deg = df["pitch"] * 180 / 3.141592653589793

# 绘图
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)

# 实验阶段背景
ax.axvspan(0, 2, color="#94a3b8", alpha=0.16, label="Stand")
ax.axvspan(2, 3, color="#f59e0b", alpha=0.16, label="Transition")
ax.axvspan(3, 8, color="#22c55e", alpha=0.10, label="Trot")

# 姿态曲线
ax.plot(time, roll_deg, color="#2563eb",
        linewidth=1.5, label="Roll", zorder=3)
ax.plot(time, pitch_deg, color="#dc2626",
        linewidth=1.5, label="Pitch", zorder=3)

# 阶段分界线
ax.axvline(2, color="#d97706", linestyle="--", linewidth=1)
ax.axvline(3, color="#16a34a", linestyle="--", linewidth=1)

# 坐标轴与标题
ax.set_title("Go2 Base Attitude During Trot Locomotion", fontsize=13)
ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Attitude (deg)", fontsize=11)
ax.set_xlim(0, 8)

# 以零度为参考线
ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="upper right", frameon=True, ncol=2)

fig.tight_layout()

# 保存图片
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, bbox_inches="tight")
plt.close(fig)

print(f"图像已保存至: {OUTPUT_PATH}")
print(f"Roll 范围: {roll_deg.min():.2f}° ~ {roll_deg.max():.2f}°")
print(f"Pitch 范围: {pitch_deg.min():.2f}° ~ {pitch_deg.max():.2f}°")
