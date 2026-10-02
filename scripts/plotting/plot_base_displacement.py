import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 路径配置
# =========================
ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = ROOT / "outputs/free_trot_v3/free_trot_v3_log.csv"
OUTPUT_PATH = ROOT / "outputs/free_trot_v3/base_displacement.png"

# =========================
# 读取数据
# =========================
df = pd.read_csv(CSV_PATH)

time = df["time"]
displacement = df["base_x"] - df["base_x"].iloc[0]

# =========================
# 绘图
# =========================
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)

# 阶段背景
ax.axvspan(0, 2, color="#94a3b8", alpha=0.16, label="Stand")
ax.axvspan(2, 3, color="#f59e0b", alpha=0.16, label="Transition")
ax.axvspan(3, 8, color="#22c55e", alpha=0.10, label="Trot")

# 位移曲线
ax.plot(
    time,
    displacement,
    color="#2563eb",
    linewidth=2.0,
    label="Base displacement",
    zorder=3,
)

# 阶段分界线
ax.axvline(2, color="#d97706", linestyle="--", linewidth=1)
ax.axvline(3, color="#16a34a", linestyle="--", linewidth=1)

# 轴标签和标题
ax.set_title("Go2 Forward Displacement During Trot Locomotion", fontsize=13)
ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Forward displacement (m)", fontsize=11)

# 坐标范围
ax.set_xlim(0, 8)
ax.set_ylim(bottom=0)

# 网格与图例
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="upper left", frameon=True, ncol=2)

fig.tight_layout()

# 保存图片
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, bbox_inches="tight")
plt.close(fig)

print(f"图像已保存至: {OUTPUT_PATH}")
print(f"最终前进位移: {displacement.iloc[-1]:.4f} m")
