import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(
    "unitree_mujoco/unitree_robots/go2/scene.xml"
)
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)

# 设置待验证的关节角
joint_names = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"]
q_test = [0.1, 0.8, -1.4]

for name, q in zip(joint_names, q_test):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qadr = model.jnt_qposadr[jid]
    data.qpos[qadr] = q

mujoco.mj_forward(model, data)

# 使用真实的机身名称
base_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
)
foot_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "FR_foot"
)

base_pos = data.xpos[base_id]
base_rot = data.xmat[base_id].reshape(3, 3)
foot_pos = data.xpos[foot_id]

# 将足端世界坐标转换到 base_link 坐标系
foot_local = base_rot.T @ (foot_pos - base_pos)

print("base_link world position:", base_pos)
print("FR_foot world position:", foot_pos)
print("FR_foot in base_link frame:", foot_local)
