import numpy as np
import mujoco

from leg_kinematics import (
    forward_kinematics,
    jacobian,
    inverse_kinematics,
)


MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
# 加载 Go2 默认站立姿态
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

legs = ["FR", "FL", "RR", "RL"]

joint_names = {
    leg: [
        f"{leg}_hip_joint",
        f"{leg}_thigh_joint",
        f"{leg}_calf_joint",
    ]
    for leg in legs
}

foot_names = {
    "FR": "FR_foot",
    "FL": "FL_foot",
    "RR": "RR_foot",
    "RL": "RL_foot",
}

base_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print("========== 四足运动学验证 ==========")

for leg in legs:
    # 读取 MuJoCo 当前关节角
    q = np.array([
        data.qpos[model.jnt_qposadr[
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
        ]]
        for name in joint_names[leg]
    ])

    # MuJoCo 足端位置转换到 base_link 坐标系
    foot_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, foot_names[leg]
    )

    base_pos = data.xpos[base_id].copy()
    base_rot = data.xmat[base_id].reshape(3, 3).copy()
    foot_pos = data.xpos[foot_id].copy()

    p_mujoco = base_rot.T @ (foot_pos - base_pos)

    # 解析 FK
    p_fk = forward_kinematics(leg, q)
    fk_error = np.linalg.norm(p_fk - p_mujoco)

    # 解析 Jacobian 与数值差分对比
    J_analytic = jacobian(leg, q)
    J_numeric = np.zeros((3, 3))
    eps = 1e-6

    for i, name in enumerate(joint_names[leg]):
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        qadr = model.jnt_qposadr[jid]

        data.qpos[qadr] += eps
        mujoco.mj_forward(model, data)

        p1 = (
            data.xmat[base_id].reshape(3, 3).T
            @ (data.xpos[foot_id] - data.xpos[base_id])
        )

        J_numeric[:, i] = (p1 - p_mujoco) / eps

        data.qpos[qadr] -= eps
        mujoco.mj_forward(model, data)

    jac_error = np.max(np.abs(J_analytic - J_numeric))

    # IK：以当前足端位置为目标，从附近姿态开始反解
    q_init = q + np.array([0.02, -0.02, 0.02])

    q_sol, ik_error, success = inverse_kinematics(
        leg, p_mujoco, q_init
    )

    print(f"\n========== {leg} ==========")
    print("关节角:", q)
    print("MuJoCo 足端位置:", p_mujoco)
    print("解析 FK 位置:", p_fk)
    print(f"FK 误差: {fk_error:.10e} m")
    print(f"Jacobian 最大误差: {jac_error:.10e}")
    print("IK 求解关节角:", q_sol)
    print(f"IK 位置误差: {ik_error:.10e} m")
    print("IK 是否收敛:", success)
