import numpy as np
import mujoco

from leg_kinematics import forward_kinematics, jacobian

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

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

# 默认站立姿态
mujoco.mj_resetDataKeyframe(model, data, 0)

# 给四条腿设置不同的非零髋关节角
hip_angles = {
    "FR": 0.15,
    "FL": -0.15,
    "RR": -0.12,
    "RL": 0.12,
}

for leg in legs:
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_names[leg][0],
    )
    data.qpos[model.jnt_qposadr[jid]] = hip_angles[leg]

mujoco.mj_forward(model, data)

print("========== 非零髋关节角验证 ==========")

for leg in legs:
    q = np.array([
        data.qpos[
            model.jnt_qposadr[
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    name,
                )
            ]
        ]
        for name in joint_names[leg]
    ])

    foot_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        foot_names[leg],
    )

    def get_foot_position():
        base_pos = data.xpos[base_id].copy()
        base_rot = data.xmat[base_id].reshape(3, 3).copy()
        foot_pos = data.xpos[foot_id].copy()
        return base_rot.T @ (foot_pos - base_pos)

    p_mujoco = get_foot_position()
    p_fk = forward_kinematics(leg, q)

    J_analytic = jacobian(leg, q)
    J_numeric = np.zeros((3, 3))

    eps = 1e-6

    for i, name in enumerate(joint_names[leg]):
        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        qadr = model.jnt_qposadr[jid]

        data.qpos[qadr] += eps
        mujoco.mj_forward(model, data)

        p1 = get_foot_position()
        J_numeric[:, i] = (p1 - p_mujoco) / eps

        data.qpos[qadr] -= eps
        mujoco.mj_forward(model, data)

    fk_error = np.linalg.norm(p_fk - p_mujoco)
    jac_error = np.max(np.abs(J_analytic - J_numeric))

    print(f"\n{leg}:")
    print("关节角:", q)
    print(f"FK 误差: {fk_error:.10e} m")
    print(f"Jacobian 最大误差: {jac_error:.10e}")
