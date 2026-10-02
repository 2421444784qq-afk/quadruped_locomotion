import csv
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. 读取数据
# ==============================

filename = "airborne_torque_tracking.csv"

with open(filename, "r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

time = np.array([float(row["time"]) for row in rows])

n = 12
q_des = np.column_stack([
    [float(row[f"q{i}_des"]) for row in rows]
    for i in range(1, n + 1)
])

q_actual = np.column_stack([
    [float(row[f"q{i}_control"]) for row in rows]
    for i in range(1, n + 1)
])

tau_raw = np.column_stack([
    [float(row[f"tau{i}_raw"]) for row in rows]
    for i in range(1, n + 1)
])

tau_applied = np.column_stack([
    [float(row[f"tau{i}_applied"]) for row in rows]
    for i in range(1, n + 1)
])

saturated = np.column_stack([
    [int(float(row[f"saturated{i}"])) for row in rows]
    for i in range(1, n + 1)
])

q_error = q_des - q_actual

# ==============================
# 2. 统计各关节力矩
# ==============================

print("\n========== 12个关节力矩统计 ==========")

for i in range(n):
    print(
        f"关节{i+1:02d}: "
        f"最大|原始力矩|={np.max(np.abs(tau_raw[:, i])):.4f} Nm, "
        f"平均|原始力矩|={np.mean(np.abs(tau_raw[:, i])):.4f} Nm, "
        f"最大|控制指令|={np.max(np.abs(tau_applied[:, i])):.4f} Nm, "
        f"限幅次数={np.sum(saturated[:, i])}"
    )

# ==============================
# 3. FR腿力矩曲线
# ==============================

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for i in range(3):
    ax = axes[i]

    ax.plot(
        time,
        tau_raw[:, i],
        label="Raw torque command",
        linewidth=1.2
    )

    ax.plot(
        time,
        tau_applied[:, i],
        "--",
        label="Applied torque command",
        linewidth=1.2
    )

    ax.set_ylabel(f"Joint {i+1} (Nm)")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1].set_xlabel("Time (s)")
fig.suptitle("FR Leg Torque Commands")
plt.tight_layout()
plt.savefig("airborne_fr_torque.png", dpi=300)
plt.show()

# ==============================
# 4. FR腿关节误差与力矩对比
# ==============================

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for i in range(3):
    ax = axes[i]

    ax.plot(
        time,
        q_error[:, i],
        label="Joint tracking error",
        color="tab:blue"
    )

    ax.set_ylabel(f"Joint {i+1} error (rad)")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(
        time,
        tau_applied[:, i],
        label="Torque command",
        color="tab:orange",
        alpha=0.7
    )
    ax2.set_ylabel("Torque (Nm)")

axes[-1].set_xlabel("Time (s)")
fig.suptitle("FR Leg Joint Error and Torque")
plt.tight_layout()
plt.savefig("airborne_fr_error_torque.png", dpi=300)
plt.show()

print("\n分析完成，已保存：")
print("1. airborne_fr_torque.png")
print("2. airborne_fr_error_torque.png")
