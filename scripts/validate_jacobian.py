import numpy as np
from kinematics import forward_kinematics, jacobian

q = np.array([0.1, 0.8, -1.4])
eps = 1e-6

J_analytic = jacobian(q)
J_numeric = np.zeros((3, 3))

for i in range(3):
    dq = np.zeros(3)
    dq[i] = eps

    p_plus = np.array(forward_kinematics(q + dq))
    p_minus = np.array(forward_kinematics(q - dq))

    J_numeric[:, i] = (p_plus - p_minus) / (2 * eps)

print("========== 解析雅可比矩阵 ==========")
print(J_analytic)

print("\n========== 数值差分雅可比矩阵 ==========")
print(J_numeric)

print("\n========== 误差 ==========")
print(J_analytic - J_numeric)

error = np.linalg.norm(J_analytic - J_numeric)
print(f"\nFrobenius norm error: {error:.12e}")
