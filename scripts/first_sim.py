import mujoco
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path("models/first_scene.xml")
data = mujoco.MjData(model)

print("模型加载成功！")
print("仿真时间步长:", model.opt.timestep)
print("初始球体高度:", data.qpos[2])

# 执行仿真
for i in range(2000):
    mujoco.mj_step(model, data)

    if i % 200 == 0:
        print(
            f"时间: {data.time:.2f}s | "
            f"球体高度: {data.qpos[2]:.4f}m"
        )

print("\n仿真完成！")
print("最终球体高度:", data.qpos[2])
print("接触数量:", data.ncon)
