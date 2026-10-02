import time
import numpy as np
import mujoco
import mujoco.viewer

from kinematics import forward_kinematics
from inverse_kinematics import inverse_kinematics
from foot_trajectory import FootTrajectory


# ==============================
# 1. 加载模型
# ==============================

model = mujoco.MjModel.from_xml_path(
    "unitree_mujoco/unitree_robots/go2/scene.xml"
)
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)


# ==============================
# 2. 关节与执行器映射
# ==============================

joint_names = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

joint_ids = [
    mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name
    )
    for name in joint_names
]

qpos_addrs = [
    model.jnt_qposadr[jid] for jid in joint_ids
]

dof_addrs = [
    model.jnt_dofadr[jid] for jid in joint_ids
]

# 站立目标姿态
q_stand = np.tile(
    np.array([0.0, 0.8, -1.4]),
    4
)

# 执行器力矩限制
torque_limits = np.array([
    23.7, 23.7, 45.43,
    23.7, 23.7, 45.43,
    23.7, 23.7, 45.43,
    23.7, 23.7, 45.43,
])


# ==============================
# 3. PD 参数
# ==============================

kp = np.tile(
    np.array([20.0, 20.0, 40.0]),
    4
)

kd = np.tile(
    np.array([1.0, 1.0, 2.0]),
    4
)


# ==============================
# 4. 轨迹参数
# ==============================

# 当前站立姿态下 FR 足端位置
p_stand = forward_kinematics(
    np.array([0.0, 0.8, -1.4])
)

trajectory = FootTrajectory(
    center=p_stand,
    step_length=0.08,
    step_height=0.05,
    duty_factor=0.6,
    period=2.0
)

# 轨迹起点
p_start = trajectory.get_position(0.0)

# 控制阶段时间
stand_transition = 1.5
foot_transition = 1.0
simulation_duration = 10.0


# ==============================
# 5. 初始状态
# ==============================

mujoco.mj_forward(model, data)

q_initial = np.array([
    data.qpos[addr] for addr in qpos_addrs
])

print("初始关节角:", q_initial)
print("站立目标角:", q_stand)
print("FR 足端站立位置:", p_stand)
print("FR 足端轨迹起点:", p_start)


# ==============================
# 6. 控制循环
# ==============================

with mujoco.viewer.launch_passive(model, data) as viewer:

    start_wall_time = time.time()

    while viewer.is_running():

        sim_time = data.time

        # 默认所有腿保持站立
        q_des = q_stand.copy()

        # 阶段一：平滑过渡到站立姿态
        if sim_time < stand_transition:

            alpha = sim_time / stand_transition
            alpha = np.clip(alpha, 0.0, 1.0)

            q_des = (
                (1.0 - alpha) * q_initial
                + alpha * q_stand
            )

        # 阶段二：FR 足端从站立位置过渡到轨迹起点
        elif sim_time < stand_transition + foot_transition:

            alpha = (
                sim_time - stand_transition
            ) / foot_transition

            alpha = np.clip(alpha, 0.0, 1.0)

            # smoothstep，降低过渡段起止处的速度突变
            alpha = alpha**2 * (3.0 - 2.0 * alpha)

            p_des = (
                (1.0 - alpha) * p_stand
                + alpha * p_start
            )

            q_fr, _, success = inverse_kinematics(
                target=p_des,
                q_init=q_des[:3],
                damping=0.05,
                max_iter=100,
                tol=1e-5
            )

            if success:
                q_des[:3] = q_fr

        # 阶段三：启动周期性足端轨迹
        else:

            trajectory_time = (
                sim_time
                - stand_transition
                - foot_transition
            )

            p_des = trajectory.get_position(
                trajectory_time
            )

            q_fr, _, success = inverse_kinematics(
                target=p_des,
                q_init=q_des[:3],
                damping=0.05,
                max_iter=100,
                tol=1e-5
            )

            if success:
                q_des[:3] = q_fr

        # 读取实际关节角和角速度
        q = np.array([
            data.qpos[addr] for addr in qpos_addrs
        ])

        dq = np.array([
            data.qvel[addr] for addr in dof_addrs
        ])

        # PD 力矩控制
        torque = kp * (q_des - q) - kd * dq

        # 力矩限幅
        torque = np.clip(
            torque,
            -torque_limits,
            torque_limits
        )

        data.ctrl[:] = torque

        # 推进物理仿真
        mujoco.mj_step(model, data)

        viewer.sync()

        # 控制仿真运行速度
        elapsed = time.time() - start_wall_time
        target_elapsed = data.time

        if target_elapsed > elapsed:
            time.sleep(target_elapsed - elapsed)

        # 每秒打印一次状态
        if int(sim_time) != int(data.time):
            print(
                f"t={data.time:.1f}s | "
                f"base_z={data.qpos[2]:.3f}m | "
                f"FR_error={np.max(np.abs(q_des[:3]-q[:3])):.3f}rad"
            )

        if data.time >= simulation_duration:
            print("仿真运行完成。")
            break
