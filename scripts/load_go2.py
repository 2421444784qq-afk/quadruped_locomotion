import mujoco
import numpy as np

# 加载 Go2 场景
model_path = "unitree_mujoco/unitree_robots/go2/scene.xml"

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 初始化仿真状态
mujoco.mj_forward(model, data)

print("=" * 50)
print("Go2 模型加载成功！")
print("=" * 50)

# 模型基本信息
print(f"仿真时间步长: {model.opt.timestep} s")
print(f"刚体数量: {model.nbody}")
print(f"关节数量: {model.njnt}")
print(f"广义坐标数量: {model.nq}")
print(f"广义速度数量: {model.nv}")
print(f"执行器数量: {model.nu}")
print(f"传感器数量: {model.nsensor}")

# 读取关节名称和初始位置
print("\n关节状态：")

for i in range(model.njnt):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i
    )

    qpos_addr = model.jnt_qposadr[i]

    if name is not None:
        print(f"{name:20s} qpos = {data.qpos[qpos_addr]:.4f}")

# 读取执行器名称
print("\n执行器列表：")

for i in range(model.nu):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i
    )

    print(f"{i:2d}: {name}")

# 机身初始位置
base_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base_link"
)

print("\n机身初始位置：")
print(data.xpos[base_id])

print("\n模型检查完成！")
