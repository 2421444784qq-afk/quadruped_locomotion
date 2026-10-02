import csv
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. 读取 CSV 数据
# ==============================

data = {}

with open("fixed_base_tracking.csv", "r", newline="") as f:
    reader = csv.DictReader(f)

    for key in reader.fieldnames:
        data[key] = []

    for row in reader:
        for key in reader.fieldnames:
            data[key].append(float(row[key]))

for key in data:
    data[key] = np.array(data[key])

time = data["time"]

# ==============================
# 2. 计算三轴跟踪误差
# ==============================

ex = data["x_actual"] - data["x_des"]
ey = data["y_actual"] - data["y_des"]
ez = data["z_actual"] - data["z_des"]

position_error = data["position_error"]

# ==============================
# 3. 输出统计指标
# ==============================

print("\n========== 足端跟踪误差统计 ==========")

for name, error in [
    ("X方向", ex),
    ("Y方向", ey),
    ("Z方向", ez),
]:
    print(
        f"{name}: "
        f"平均绝对误差={np.mean(np.abs(error)):.6f} m, "
        f"RMSE={np.sqrt(np.mean(error**2)):.6f} m, "
        f"最大绝对误差={np.max(np.abs(error)):.6f} m"
    )

print(
    f"\n三维位置误差: "
    f"平均={np.mean(position_error):.6f} m, "
    f"RMSE={np.sqrt(np.mean(position_error**2)):.6f} m, "
    f"最大={np.max(position_error):.6f} m"
)

print("\n========== 关节角度误差统计 ==========")

for i in range(1, 4):
    error = data[f"q{i}_actual"] - data[f"q{i}_des"]

    print(
        f"关节 q{i}: "
        f"平均绝对误差={np.mean(np.abs(error)):.6f} rad, "
        f"最大绝对误差={np.max(np.abs(error)):.6f} rad"
    )

# ==============================
# 4. 绘制足端三维位置跟踪曲线
# ==============================

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for ax, axis, label in zip(
    axes,
    ["x", "y", "z"],
    ["X", "Y", "Z"]
):
    ax.plot(
        time,
        data[f"{axis}_des"],
        label="Desired",
        linewidth=1.5
    )

    ax.plot(
        time,
        data[f"{axis}_actual"],
        label="Actual",
        linewidth=1.2,
        linestyle="--"
    )

    ax.set_ylabel(f"{label} (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1].set_xlabel("Time (s)")
fig.suptitle("Foot Position Tracking")
fig.tight_layout()
fig.savefig("foot_position_tracking.png", dpi=200)
plt.close(fig)

# ==============================
# 5. 绘制三轴误差曲线
# ==============================

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(time, ex, label="X error")
ax.plot(time, ey, label="Y error")
ax.plot(time, ez, label="Z error")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position Error (m)")
ax.set_title("Foot Position Tracking Error")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig("foot_position_error.png", dpi=200)
plt.close(fig)

# ==============================
# 6. 绘制三维位置误差
# ==============================

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(time, position_error, label="Position error")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Error (m)")
ax.set_title("3D Foot Position Error")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig("foot_position_error_3d.png", dpi=200)
plt.close(fig)

# ==============================
# 7. 绘制关节角度跟踪曲线
# ==============================

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for i, ax in enumerate(axes, start=1):
    ax.plot(
        time,
        data[f"q{i}_des"],
        label="Desired",
        linewidth=1.5
    )

    ax.plot(
        time,
        data[f"q{i}_actual"],
        label="Actual",
        linewidth=1.2,
        linestyle="--"
    )

    ax.set_ylabel(f"q{i} (rad)")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1].set_xlabel("Time (s)")
fig.suptitle("Joint Angle Tracking")
fig.tight_layout()
fig.savefig("joint_angle_tracking.png", dpi=200)
plt.close(fig)

print("\n图像已保存：")
print("1. foot_position_tracking.png")
print("2. foot_position_error.png")
print("3. foot_position_error_3d.png")
print("4. joint_angle_tracking.png")
