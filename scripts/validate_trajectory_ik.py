import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inverse_kinematics import inverse_kinematics
from kinematics import forward_kinematics


# 读取足端轨迹
data = np.loadtxt(
    "foot_trajectory.csv",
    delimiter=",",
    skiprows=1
)

times = data[:, 0]
targets = data[:, 1:4]

# 初始关节角：之前验证通过的站立姿态
q_init = np.array([0.0, 0.8, -1.4])

# Go2 FR 腿关节限位（弧度）
q_min = np.array([-1.0472, -1.5708, -2.7227])
q_max = np.array([ 1.0472,  3.4907, -0.83776])

joint_names = ["Hip", "Thigh", "Calf"]

solutions = []
errors = []
success_flags = []

q_current = q_init.copy()

for target in targets:
    q_sol, error, success = inverse_kinematics(
        target=target,
        q_init=q_current,
        damping=0.05,
        max_iter=100,
        tol=1e-5
    )

    solutions.append(q_sol)
    errors.append(error)
    success_flags.append(success)

    # 连续跟踪：将本次解作为下一次迭代的初始值
    q_current = q_sol.copy()

solutions = np.array(solutions)
errors = np.array(errors)
success_flags = np.array(success_flags)

# 检查关节限位
within_limits = np.all(
    (solutions >= q_min) & (solutions <= q_max),
    axis=1
)

print("========== 轨迹 IK 可行性验证 ==========")
print(f"轨迹点总数: {len(targets)}")
print(f"成功收敛点数: {np.sum(success_flags)}")
print(f"关节限位内点数: {np.sum(within_limits)}")
print(f"最大足端误差: {np.max(errors):.8e} m")
print(f"平均足端误差: {np.mean(errors):.8e} m")
print(f"最大单步关节角变化: {np.max(np.abs(np.diff(solutions, axis=0))):.6f} rad")

if np.all(success_flags) and np.all(within_limits):
    print("\n结果：整条轨迹通过 IK 和关节限位检查。")
else:
    print("\n结果：存在未收敛点或关节超限，需要进一步检查。")

# 绘制关节角曲线
plt.figure(figsize=(10, 6))

for i, name in enumerate(joint_names):
    plt.plot(times, solutions[:, i], label=name)

plt.xlabel("Time (s)")
plt.ylabel("Joint angle (rad)")
plt.title("Go2 FR Joint Trajectories")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("joint_trajectory.png", dpi=200)
plt.show()

# 绘制足端位置误差
plt.figure(figsize=(10, 4))
plt.plot(times, errors)
plt.xlabel("Time (s)")
plt.ylabel("Position error (m)")
plt.title("Foot Position Tracking Error (IK)")
plt.grid(True)
plt.tight_layout()
plt.savefig("ik_error.png", dpi=200)
plt.show()
