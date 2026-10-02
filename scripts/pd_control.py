import time
import mujoco
import mujoco.viewer
import numpy as np

# 加载 Go2 模型
MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print("已加载官方 home 关键帧")
print("初始机身位置:", data.qpos[:3])

# 仿真初始化


# PD 控制参数
# 按关节类型设置 PD 参数
KP_HIP = 20.0
KD_HIP = 1.0

KP_THIGH = 20.0
KD_THIGH = 1.0

KP_CALF = 40.0
KD_CALF = 2.0

# 记录每个执行器对应的关节索引
qpos_ids = []
dof_ids = []

for i in range(model.nu):
    joint_id = model.actuator_trnid[i, 0]

    qpos_ids.append(model.jnt_qposadr[joint_id])
    dof_ids.append(model.jnt_dofadr[joint_id])

qpos_ids = np.array(qpos_ids)
dof_ids = np.array(dof_ids)

# 将初始关节角度作为目标角度
# Go2 四足机器人的初始站立目标姿态
# 执行器顺序：FR、FL、RR、RL
# 直接使用官方 home 关键帧中的关节角度
# Unitree Go2 站立参考姿态
# 执行器顺序：FR、FL、RR、RL
q_des = np.array([
    0.0, 0.8, -1.4,
    0.0, 0.8, -1.4,
    0.0, 0.8, -1.4,
    0.0, 0.8, -1.4
])

print("目标关节角度:", q_des)

print("开始 PD 控制仿真……")

# 启动可视化
# 记录初始关节角度
q_start = data.qpos[qpos_ids].copy()

# 目标姿态过渡时间
transition_time = 1.0

with mujoco.viewer.launch_passive(model, data) as viewer:

    start_time = time.time()
    last_print_time = start_time

    while viewer.is_running():

        step_start = time.time()

        # 当前关节位置和速度
        q = data.qpos[qpos_ids]
        qd = data.qvel[dof_ids]

        # PD 控制律
        alpha = min(data.time / transition_time, 1.0)
        q_target = (1 - alpha) * q_start + alpha * q_des
        torque = np.zeros(model.nu)

        for i in range(model.nu):

            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                i
            )

            if "hip" in name:
                kp = KP_HIP
                kd = KD_HIP

            elif "thigh" in name:
                kp = KP_THIGH
                kd = KD_THIGH

            elif "calf" in name:
                kp = KP_CALF
                kd = KD_CALF

            torque[i] = kp * (q_target[i] - q[i]) - kd * qd[i]

        # 根据执行器力矩范围进行限制
        for i in range(model.nu):
            low, high = model.actuator_ctrlrange[i]
            data.ctrl[i] = np.clip(torque[i], low, high)

        # 执行一步物理仿真
        mujoco.mj_step(model, data)

        # 同步可视化
        viewer.sync()

        # 每秒打印一次状态
        now = time.time()
        if now - last_print_time >= 1.0:
            elapsed = now - start_time
            error = np.max(np.abs(q_des - q))

            print(
                f"\n仿真时间: {data.time:.2f}s | "
                f"最大关节误差: {error:.4f} rad | "
                f"机身高度: {data.qpos[2]:.4f} m"
            )

            print("各关节状态：")

            for i in range(model.nu):
                name = mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    i
                )

                print(
                    f"{name:12s} | "
                    f"目标: {q_target[i]:7.3f} | "
                    f"实际: {q[i]:7.3f} | "
                    f"误差: {q_target[i] - q[i]:7.3f} | "
                    f"力矩: {data.ctrl[i]:7.3f}"
                )

            last_print_time = now

        # 按仿真时间步长控制运行速度
        remaining = model.opt.timestep - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)