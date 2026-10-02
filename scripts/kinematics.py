import numpy as np
from math import sin, cos

# Go2 FR 腿参数
L1 = 0.0955
L2 = 0.213
L3 = 0.213

# FR 髋关节相对于 base_link 的固定偏移
HIP_OFFSET = np.array([0.1934, -0.0465, 0.0])


def forward_kinematics(q):
    """
    Go2 FR 腿正运动学

    输入:
        q = [q1, q2, q3]
        分别为髋外展、大腿、小腿关节角

    输出:
        足端刚体原点在 base_link 坐标系下的位置 [x, y, z]
    """
    q1, q2, q3 = q

    x = -L2 * sin(q2) - L3 * sin(q2 + q3)

    y = (
        -L1 * cos(q1)
        + L2 * sin(q1) * cos(q2)
        + L3 * sin(q1) * cos(q2 + q3)
    )

    z = (
        -L1 * sin(q1)
        - L2 * cos(q1) * cos(q2)
        - L3 * cos(q1) * cos(q2 + q3)
    )

    return HIP_OFFSET + np.array([x, y, z])


def jacobian(q):
    """
    Go2 FR 腿解析雅可比矩阵

    输入:
        q = [q1, q2, q3]

    输出:
        3x3 Jacobian，满足：
        foot_velocity = J(q) @ joint_velocity
    """
    q1, q2, q3 = q

    s1, c1 = sin(q1), cos(q1)
    s2, c2 = sin(q2), cos(q2)
    s23, c23 = sin(q2 + q3), cos(q2 + q3)

    J = np.array([
        [
            0.0,
            -L2 * c2 - L3 * c23,
            -L3 * c23
        ],
        [
            L1 * s1 + L2 * c1 * c2 + L3 * c1 * c23,
            -L2 * s1 * s2 - L3 * s1 * s23,
            -L3 * s1 * s23
        ],
        [
            -L1 * c1 + L2 * s1 * c2 + L3 * s1 * c23,
            L2 * c1 * s2 + L3 * c1 * s23,
            L3 * c1 * s23
        ]
    ])

    return J
