import numpy as np

from kinematics import forward_kinematics, jacobian


def inverse_kinematics(
    target,
    q_init,
    damping=0.05,
    max_iter=100,
    tol=1e-5,
    max_step=0.2
):
    """
    基于阻尼最小二乘法的 Go2 单腿逆运动学。

    参数:
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
    target = np.asarray(target, dtype=float)
    q = np.asarray(q_init, dtype=float).copy()

    for iteration in range(max_iter):
        current = np.asarray(forward_kinematics(q))
        error = target - current

        error_norm = np.linalg.norm(error)

        if error_norm < tol:
            return q, error_norm, True

        J = jacobian(q)

        dq = J.T @ np.linalg.solve(
            J @ J.T + damping**2 * np.eye(3),
            error
        )

        # 限制单次更新幅度，避免迭代跳跃
        dq_norm = np.linalg.norm(dq)
        if dq_norm > max_step:
            dq *= max_step / dq_norm

        q += dq

    final_error = np.linalg.norm(
        target - forward_kinematics(q)
    )

    return q, final_error, final_error < tol


if __name__ == "__main__":
    # 选取一个已知关节姿态作为测试
    q_true = np.array([0.1, 0.8, -1.4])

    # 通过正运动学生成目标足端位置
    target = forward_kinematics(q_true)

    # 使用不同于真实姿态的初始值
    q_init = np.array([0.0, 0.6, -1.2])

    q_sol, error, success = inverse_kinematics(
        target,
        q_init
    )

    print("========== DLS 逆运动学测试 ==========")
    print("目标足端位置:", target)
    print("初始关节角:", q_init)
    print("求解关节角:", q_sol)
    print("真实关节角:", q_true)
    print("最终位置误差:", error)
    print("是否收敛:", success)
    print("反解足端位置:", forward_kinematics(q_sol))
