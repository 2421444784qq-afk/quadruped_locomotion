import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path(
    "unitree_mujoco/unitree_robots/go2/scene.xml"
)
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

legs = ["FR", "FL", "RR", "RL"]
joint_types = ["hip", "thigh", "calf"]

joint_names = {
    leg: [
        f"{leg}_hip_joint",
        f"{leg}_thigh_joint",
        f"{leg}_calf_joint",
    ]
    for leg in legs
}

foot_ids = {
    leg: mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        f"{leg}_foot"
    )
    for leg in legs
}

base_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base_link"
)


def get_foot_position(leg):
    base_pos = data.xpos[base_id].copy()
    base_rot = data.xmat[base_id].reshape(3, 3).copy()

    foot_pos = data.xpos[foot_ids[leg]].copy()

    return base_rot.T @ (foot_pos - base_pos)


print("========== 四腿关节扰动实验 ==========")

for leg in legs:

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    print(f"\n========== {leg} ==========")

    p0 = get_foot_position(leg)

    print("初始足端位置:", p0)

    for joint_name, joint_type in zip(
        joint_names[leg], joint_types
    ):

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name
        )

        qadr = model.jnt_qposadr[jid]

        # 在当前关节角基础上增加 0.01 rad
        data.qpos[qadr] += 0.01

        mujoco.mj_forward(model, data)

        p1 = get_foot_position(leg)

        dp = (p1 - p0) / 0.01

        print(
            f"{joint_type:>6s}: "
            f"dx/dq={dp[0]: .6f}, "
            f"dy/dq={dp[1]: .6f}, "
            f"dz/dq={dp[2]: .6f}"
        )

        # 恢复该关节角
        data.qpos[qadr] -= 0.01
        mujoco.mj_forward(model, data)
