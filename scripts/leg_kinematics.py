import numpy as np
from math import sin, cos


# Go2 几何参数
L1 = 0.0955
L2 = 0.213
L3 = 0.213

# 各腿髋关节相对于 base_link 的固定偏移
LEG_CONFIG = {
    "FR": {
        "hip_offset": np.array([0.1934, -0.0465, 0.0]),
        "side": -1.0,
    },
    "FL": {
        "hip_offset": np.array([0.1934, 0.0465, 0.0]),
        "side": 1.0,
    },
    "RR": {
        "hip_offset": np.array([-0.1934, -0.0465, 0.0]),
        "side": -1.0,
    },
    "RL": {
        "hip_offset": np.array([-0.1934, 0.0465, 0.0]),
        "side": 1.0,
    },
}


def _check_leg(leg):
    if leg not in LEG_CONFIG:
        raise ValueError(
            f"未知腿名: {leg}，可选值为 {list(LEG_CONFIG.keys())}"
        )


def forward_kinematics(leg, q):
    """
    四足机器人统一正运动学。

    参数:
        leg: "FR", "FL", "RR", "RL"
        q: [q1, q2, q3]
           分别为髋外展、大腿、小腿关节角

    返回:
        足端在 base_link 坐标系下的位置 [x, y, z]
    """
    _check_leg(leg)

    q1, q2, q3 = np.asarray(q, dtype=float)
    side = LEG_CONFIG[leg]["side"]
    hip_offset = LEG_CONFIG[leg]["hip_offset"]

    s1, c1 = sin(q1), cos(q1)
    s2, c2 = sin(q2), cos(q2)
    s23, c23 = sin(q2 + q3), cos(q2 + q3)

    x = -L2 * s2 - L3 * s23

    y = (
        side * L1 * c1
        + L2 * s1 * c2
        + L3 * s1 * c23
    )

    z = (
        side * L1 * s1
        - L2 * c1 * c2
        - L3 * c1 * c23
    )

    return hip_offset + np.array([x, y, z])


def jacobian(leg, q):
    """
    四足机器人统一解析雅可比矩阵。

    满足:
        foot_velocity = J(q) @ joint_velocity
    """
    _check_leg(leg)

    q1, q2, q3 = np.asarray(q, dtype=float)
    side = LEG_CONFIG[leg]["side"]

    s1, c1 = sin(q1), cos(q1)
    s2, c2 = sin(q2), cos(q2)
    s23, c23 = sin(q2 + q3), cos(q2 + q3)

    J = np.array([
        [
            0.0,
            -L2 * c2 - L3 * c23,
            -L3 * c23,
        ],
        [
            -side * L1 * s1
            + L2 * c1 * c2
            + L3 * c1 * c23,
            -L2 * s1 * s2 - L3 * s1 * s23,
            -L3 * s1 * s23,
        ],
        [
            side * L1 * c1
            + L2 * s1 * c2
            + L3 * s1 * c23,
            L2 * c1 * s2
            + L3 * c1 * s23,
            L3 * c1 * s23,
        ],
    ])

    return J


def inverse_kinematics(
    leg,
    target,
    q_init,
    damping=0.05,
    max_iter=100,
    tol=1e-5,
    max_step=0.2,
):
    """
    基于阻尼最小二乘法的四足统一逆运动学。

    参数:
        leg: 腿名
        target: 目标足端位置，base_link 坐标系
        q_init: 初始关节角
        damping: 阻尼系数
        max_iter: 最大迭代次数
        tol: 位置误差阈值
        max_step: 单次迭代最大关节角变化量

    返回:
        q: 求解得到的关节角
        error: 最终足端位置误差
        success: 是否收敛
    """
    _check_leg(leg)

    target = np.asarray(target, dtype=float)
    q = np.asarray(q_init, dtype=float).copy()

    for _ in range(max_iter):
        current = forward_kinematics(leg, q)
        error = target - current
        error_norm = np.linalg.norm(error)

        if error_norm < tol:
            return q, error_norm, True

        J = jacobian(leg, q)

        dq = J.T @ np.linalg.solve(
            J @ J.T + damping**2 * np.eye(3),
            error,
        )

        dq_norm = np.linalg.norm(dq)

        if dq_norm > max_step:
            dq *= max_step / dq_norm

        q += dq

    final_error = np.linalg.norm(
        target - forward_kinematics(leg, q)
    )

    return q, final_error, final_error < tol