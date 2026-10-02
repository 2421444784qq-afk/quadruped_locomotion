import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path(
    "unitree_mujoco/unitree_robots/go2/scene.xml"
)

data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

base_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base_link"
)

base_pos = data.xpos[base_id]
base_rot = data.xmat[base_id].reshape(3, 3)

foot_names = [
    "FR_foot",
    "FL_foot",
    "RR_foot",
    "RL_foot"
]

print("========== Go2 四足站立位置 ==========")

centers = {}

for name in foot_names:

    foot_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name
    )

    foot_world = data.xpos[foot_id]

    foot_base = base_rot.T @ (
        foot_world - base_pos
    )

    leg_name = name.split("_")[0]

    centers[leg_name] = foot_base

    print(
        f"{leg_name}: "
        f"{np.array2string(foot_base, precision=6)}"
    )

print("\n========== Python 字典 ==========")
print("centers = {")

for name, pos in centers.items():
    print(
        f'    "{name}": np.array({pos.tolist()}),'
    )

print("}")
