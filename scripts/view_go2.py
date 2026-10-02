import time
import mujoco
import mujoco.viewer

# 加载 Go2 场景
model_path = "unitree_mujoco/unitree_robots/go2/scene.xml"

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 初始化物理状态
mujoco.mj_forward(model, data)

# 获取 Go2 基座 body 的 id（通过名字查找）
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

print("Go2 可视化仿真启动！")
print("当前为无控制输入的自由动力学仿真。")
print("关闭窗口即可退出。")

# 启动被动式 Viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    # 可选：初始化一下相机参数
    viewer.cam.distance = 2.0

    while viewer.is_running():

        step_start = time.time()

        # 推进一个仿真步
        mujoco.mj_step(model, data)

        # 让相机跟随 Go2 的 base
        viewer.cam.lookat[:] = data.xpos[base_id]
        viewer.cam.distance = 2.0

        # 同步仿真状态到窗口
        viewer.sync()

        # 尽量按照仿真时间步长运行
        elapsed = time.time() - step_start
        time_until_next_step = model.opt.timestep - elapsed

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)