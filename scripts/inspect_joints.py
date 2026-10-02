import mujoco
import numpy as np

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)

print("\n========== 关节定义 ==========")

for i in range(model.njnt):
    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_JOINT, i
    )

    joint_type = model.jnt_type[i]
    axis = model.jnt_axis[i]
    limited = model.jnt_limited[i]
    joint_range = model.jnt_range[i]

    print(f"\n关节: {name}")
    print(f"  类型: {joint_type}")
    print(f"  旋转轴: {axis}")
    print(f"  是否有限位: {limited}")

    if limited:
        print(
            f"  角度范围: "
            f"[{joint_range[0]:.3f}, {joint_range[1]:.3f}] rad"
        )

print("\n========== 执行器定义 ==========")

for i in range(model.nu):
    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, i
    )

    joint_id = model.actuator_trnid[i, 0]
    joint_name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
    )

    print(f"\n执行器 {i}: {name}")
    print(f"  控制关节: {joint_name}")
    print(f"  力矩范围: {model.actuator_ctrlrange[i]}")

print("\n========== 模型初始状态 ==========")

data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print("机身位置:", data.qpos[:3])
print("机身四元数:", data.qpos[3:7])

for i in range(model.njnt):
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, i
        )
        qpos_id = model.jnt_qposadr[i]
        print(f"{name:20s}: {data.qpos[qpos_id]:.4f} rad")
