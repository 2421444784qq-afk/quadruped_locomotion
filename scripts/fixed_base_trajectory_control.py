import time
import csv
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
# 2. 关节映射
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

q_stand = np.tile(
    np.array([0.0, 0.8, -1.4]),
    4
)

kp = np.tile(
    np.array([20.0, 20.0, 40.0]),
    4
)

kd = np.tile(
    np.array([1.0, 1.0, 2.0]),
    4
)

torque_limits = np.tile(
    np.array([23.7, 23.7, 45.43]),
    4
)


# ==============================
# 3. 找到根部自由关节
# ==============================

free_joint_ids = np.where(
    model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
)[0]

if len(free_joint_ids) != 1:
    raise RuntimeError(
        f"预期一个 freejoint，实际找到 {len(free_joint_ids)} 个"
    )

free_id = free_joint_ids[0]
root_qpos_adr = model.jnt_qposadr[free_id]
root_dof_adr = model.jnt_dofadr[free_id]

print("根部 freejoint ID:", free_id)
print("根部 qpos 地址:", root_qpos_adr)
print("根部 dof 地址:", root_dof_adr)


# ==============================
# 4. 足端轨迹
# ==============================

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

p_start = trajectory.get_position(0.0)


# ==============================
# 5. 控制参数
# ==============================

settle_time = 2.0
transition_time = 1.0
trajectory_duration = 6.0

simulation_duration = (
    settle_time + transition_time + trajectory_duration
)


# ==============================
# 6. 控制函数
# ==============================

def get_joint_state():
    q = np.array([
        data.qpos[adr] for adr in qpos_addrs
    ])

    dq = np.array([
        data.qvel[adr] for adr in dof_addrs
    ])

    return q, dq


def apply_pd(q_des):
    q, dq = get_joint_state()

    torque = kp * (q_des - q) - kd * dq
    torque = np.clip(
        torque,
        -torque_limits,
        torque_limits
    )

    data.ctrl[:] = torque


def get_foot_position_base():
    base_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
    )

    foot_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "FR_foot"
    )

    base_pos = data.xpos[base_id]
    base_rot = data.xmat[base_id].reshape(3, 3)
    foot_pos = data.xpos[foot_id]

    return base_rot.T @ (foot_pos - base_pos)


# ==============================
# 7. 仿真主循环
# ==============================

with mujoco.viewer.launch_passive(model, data) as viewer:

    start_wall_time = time.time()

    root_pose_locked = None
    log_data = []

    while viewer.is_running():

        sim_time = data.time

        q, dq = get_joint_state()
        q_des = q_stand.copy()

        # 阶段一：站立并等待机身稳定
        if sim_time < settle_time:

            apply_pd(q_des)
            mujoco.mj_step(model, data)

        else:

            # 首次进入固定机身阶段时，保存根部位姿
            if root_pose_locked is None:

                root_pose_locked = data.qpos[
                    root_qpos_adr:root_qpos_adr + 7
                ].copy()

                print(
                    "\n机身已锁定，根部位置:",
                    root_pose_locked[:3]
                )

            # 阶段二：FR 足端平滑过渡到轨迹起点
            if sim_time < settle_time + transition_time:

                alpha = (
                    sim_time - settle_time
                ) / transition_time

                alpha = np.clip(alpha, 0.0, 1.0)
                alpha = alpha**2 * (3.0 - 2.0 * alpha)

                q_now = q[:3]

                p_now = forward_kinematics(q_now)

                p_des = (
                    (1.0 - alpha) * p_now
                    + alpha * p_start
                )

            # 阶段三：周期性轨迹跟踪
            else:

                trajectory_time = (
                    sim_time
                    - settle_time
                    - transition_time
                )

                p_des = trajectory.get_position(
                    trajectory_time
                )

            # DLS 逆运动学
            q_fr, ik_error, success = inverse_kinematics(
                target=p_des,
                q_init=q[:3],
                damping=0.05,
                max_iter=100,
                tol=1e-5
            )

            if success:
                q_des[:3] = q_fr

            # PD 控制
            apply_pd(q_des)

            mujoco.mj_step(model, data)

            # 锁定根部位置与姿态，并清零根部速度
            data.qpos[
                root_qpos_adr:root_qpos_adr + 7
            ] = root_pose_locked

            data.qvel[
                root_dof_adr:root_dof_adr + 6
            ] = 0.0

            mujoco.mj_forward(model, data)

            # 记录实际足端位置与误差
            p_actual = get_foot_position_base()

            position_error = np.linalg.norm(
                p_des - p_actual
            )

            q_actual, _ = get_joint_state()

            joint_error = np.max(
                np.abs(q_des[:3] - q_actual[:3])
            )

            log_data.append([
                sim_time,
                *p_des,
                *p_actual,
                position_error,
                joint_error,
                *q_des[:3],
                *q_actual[:3]
            ])

        viewer.sync()

        elapsed = time.time() - start_wall_time
        if data.time > elapsed:
            time.sleep(data.time - elapsed)

        if data.time >= simulation_duration:
            print("\n仿真完成。")
            break

    # 保存数据
    if log_data:
        header = [
            "time",
            "x_des", "y_des", "z_des",
            "x_actual", "y_actual", "z_actual",
            "position_error",
            "joint_error",
            "q1_des", "q2_des", "q3_des",
            "q1_actual", "q2_actual", "q3_actual"
        ]

        with open(
            "fixed_base_tracking.csv",
            "w",
            newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(log_data)

        errors = np.array(log_data)[:, 7]

        print("数据已保存：fixed_base_tracking.csv")
        print(f"最大足端位置误差：{np.max(errors):.6f} m")
        print(f"平均足端位置误差：{np.mean(errors):.6f} m")
