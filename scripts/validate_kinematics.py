import mujoco
import numpy as np

from kinematics import forward_kinematics

model = mujoco.MjModel.from_xml_path(
    "unitree_mujoco/unitree_robots/go2/scene.xml"
)
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)

base_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
)
foot_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "FR_foot"
)

joint_names = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint"
]

joint_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    for name in joint_names
]

qpos_addrs = [model.jnt_qposadr[jid] for jid in joint_ids]

test_cases = [
    [0.1, 0.8, -1.4],
    [0.0, 0.8, -1.4],
    [0.2, 1.0, -1.5],
    [-0.2, 0.6, -1.2],
    [0.3, 1.2, -1.8],
]

print("========== 正运动学验证 ==========")

for q in test_cases:
    mujoco.mj_resetDataKeyframe(model, data, 0)

    for addr, angle in zip(qpos_addrs, q):
        data.qpos[addr] = angle

    mujoco.mj_forward(model, data)

    base_pos = data.xpos[base_id]
    base_rot = data.xmat[base_id].reshape(3, 3)
    foot_pos = data.xpos[foot_id]

    p_mujoco = base_rot.T @ (foot_pos - base_pos)
    p_analytic = np.array(forward_kinematics(q))

    error = np.linalg.norm(p_mujoco - p_analytic)

    print(f"\nq = {q}")
    print("MuJoCo :", np.round(p_mujoco, 6))
    print("解析解 :", np.round(p_analytic, 6))
    print(f"误差   : {error:.10f} m")

