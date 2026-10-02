import os
import time
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer

from leg_kinematics import forward_kinematics, inverse_kinematics

from controllers.v3_model_based.state_estimator import StateEstimator
from controllers.v3_model_based.foot_trajectory_generator import (
    FootTrajectoryGenerator,
)


# ============================================================
# 1. 实验配置
# ============================================================

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
OUTPUT_DIR = "outputs/free_trot_v3_transition"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEGS = ["FR", "FL", "RR", "RL"]
JOINT_TYPES = ["hip", "thigh", "calf"]

JOINT_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in LEGS
    for joint in JOINT_TYPES
]

ACTUATOR_NAMES = [
    f"{leg}_{joint}"
    for leg in LEGS
    for joint in JOINT_TYPES
]

Q_STAND_LEG = np.array([0.0, 0.8, -1.45])
Q_STAND = np.tile(Q_STAND_LEG, 4)

KP = np.tile([20.0, 20.0, 40.0], 4)
KD = np.tile([1.0, 1.0, 2.0], 4)

# 步态参数
STEP_LENGTH = 0.08
STEP_HEIGHT = 0.05
DUTY_FACTOR = 0.6
GAIT_PERIOD = 0.5

# 状态反馈参数
DESIRED_VELOCITY_X = 0.10
FEEDBACK_GAIN = 0.15
MAX_FOOTHOLD_OFFSET = 0.05

# 仿真阶段
SETTLE_TIME = 2.0
TRANSITION_TIME = 1.0
TROT_DURATION = 5.0
# Trot启动阶段：从对称站立足端位置逐渐过渡到完整步态
GAIT_RAMP_TIME = 1.0

SIM_DURATION = SETTLE_TIME + TRANSITION_TIME + TROT_DURATION


# ============================================================
# 2. 加载模型
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print("模型加载成功")
print(f"仿真时间步长: {model.opt.timestep:.6f} s")


# ============================================================
# 3. 获取关节与执行器索引
# ============================================================

qpos_ids = []
qvel_ids = []
actuator_ids = []

for joint_name, actuator_name in zip(JOINT_NAMES, ACTUATOR_NAMES):
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
    )
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
    )

    if joint_id < 0 or actuator_id < 0:
        raise ValueError(
            f"未找到关节或执行器: {joint_name}, {actuator_name}"
        )

    qpos_ids.append(model.jnt_qposadr[joint_id])
    qvel_ids.append(model.jnt_dofadr[joint_id])
    actuator_ids.append(actuator_id)

qpos_ids = np.array(qpos_ids)
qvel_ids = np.array(qvel_ids)
actuator_ids = np.array(actuator_ids)

ctrl_limits = model.actuator_ctrlrange[actuator_ids]

if len(actuator_ids) != 12:
    raise RuntimeError("当前脚本要求模型具有12个驱动关节")


# ============================================================
# 4. 获取机身及足端 ID
# ============================================================

base_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
)

foot_ids = {
    leg: mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot"
    )
    for leg in LEGS
}

if base_id < 0 or any(body_id < 0 for body_id in foot_ids.values()):
    raise RuntimeError("未找到机身或足端 body")


# ============================================================
# 5. 初始化状态估计器
# ============================================================

state_estimator = StateEstimator(
    model,
    data,
    body_name="base_link",
)


# ============================================================
# 6. 初始化站立姿态与足端中心
# ============================================================

data.qpos[qpos_ids] = Q_STAND
data.qvel[qvel_ids] = 0.0
data.ctrl[:] = 0.0

mujoco.mj_forward(model, data)

centers = {
    leg: forward_kinematics(leg, Q_STAND_LEG)
    for leg in LEGS
}

for leg in LEGS:
    print(f"{leg} 站立足端中心: {centers[leg]}")


# ============================================================
# 7. 初始化 V3 足端轨迹生成器
# ============================================================

foot_generator = FootTrajectoryGenerator(
    period=GAIT_PERIOD,
    duty_factor=DUTY_FACTOR,
    step_length=STEP_LENGTH,
    clearance=STEP_HEIGHT,
    foot_centers=centers,
    feedback_gain=FEEDBACK_GAIN,
    max_foothold_offset=MAX_FOOTHOLD_OFFSET,
)

print("\nV3 状态反馈步态控制器初始化完成")


# ============================================================
# 8. 状态读取
# ============================================================

def get_joint_state():
    return data.qpos[qpos_ids].copy(), data.qvel[qvel_ids].copy()


def get_base_state():
    position = data.xpos[base_id].copy()
    rotation = data.xmat[base_id].reshape(3, 3).copy()
    quaternion = data.xquat[base_id].copy()
    return position, rotation, quaternion


def get_foot_positions_world():
    return {
        leg: data.xpos[foot_ids[leg]].copy()
        for leg in LEGS
    }


def get_foot_positions_base():
    base_pos, base_rot, _ = get_base_state()
    result = {}

    for leg in LEGS:
        foot_pos = data.xpos[foot_ids[leg]]
        result[leg] = base_rot.T @ (foot_pos - base_pos)

    return result


# ============================================================
# 9. PD + 重力补偿
# ============================================================

def apply_control(q_des):
    q, dq = get_joint_state()

    torque_pd = KP * (q_des - q) - KD * dq

    qvel_backup = data.qvel.copy()

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    torque_gravity = data.qfrc_bias[qvel_ids].copy()

    data.qvel[:] = qvel_backup
    mujoco.mj_forward(model, data)

    torque_raw = torque_pd + torque_gravity

    torque_applied = np.clip(
        torque_raw,
        ctrl_limits[:, 0],
        ctrl_limits[:, 1],
    )

    data.ctrl[actuator_ids] = torque_applied

    saturated = np.abs(torque_raw - torque_applied) > 1e-6

    return q, dq, torque_raw, torque_applied, saturated


# ============================================================
# 10. 仿真与数据记录
# ============================================================

records = []
start_wall_time = time.time()

print("\n开始 V3 自由机身 Trot 仿真")
print(f"总时长: {SIM_DURATION:.2f} s")
print(f"期望前向速度: {DESIRED_VELOCITY_X:.3f} m/s")

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        t = data.time
        q, dq = get_joint_state()

        q_des = Q_STAND.copy()
        phase = "stand"
        gait_time = 0.0
        target_dict = {}

        # ----------------------------------------------------
        # 阶段一：自由机身站立
        # ----------------------------------------------------

        if t < SETTLE_TIME:
            phase = "stand"

        # ----------------------------------------------------
        # 阶段二：平滑过渡
        # ----------------------------------------------------

        elif t < SETTLE_TIME + TRANSITION_TIME:
            phase = "transition"

            alpha = (
                t - SETTLE_TIME
            ) / TRANSITION_TIME

            alpha = np.clip(alpha, 0.0, 1.0)
            alpha = alpha**2 * (3.0 - 2.0 * alpha)

            for i, leg in enumerate(LEGS):
                q_now = q[3*i:3*i+3]

                # 当前实际足端位置
                p_now = forward_kinematics(leg, q_now)

                # 对称站立足端中心，不提前引入步态轨迹
                p_target = centers[leg]

                # Smoothstep平滑过渡
                p_des = (1.0 - alpha) * p_now + alpha * p_target

                q_ik, error, success = inverse_kinematics(
                    leg=leg,
                    target=p_des,
                    q_init=q_now,
                    damping=0.05,
                    max_iter=100,
                    tol=1e-5,
                )

                if success:
                    q_des[3*i:3*i+3] = q_ik

        # ----------------------------------------------------
        # 阶段三：状态反馈 Trot
        # ----------------------------------------------------

        else:
            phase = "trot"
            gait_time = t - SETTLE_TIME - TRANSITION_TIME

            # 读取机身状态
            state = state_estimator.get_state()

            rotation = state["rotation_world"]
            velocity_world = state["velocity_world"]

            # 将世界系速度转换到机身坐标系
            velocity_body = rotation.T @ velocity_world
            body_velocity_x = float(velocity_body[0])

            # 根据机身实际速度生成动态足端目标
            target_dict = foot_generator.get_targets(
                time=gait_time,
                body_velocity_x=body_velocity_x,
                desired_velocity_x=DESIRED_VELOCITY_X,
                feedback_enabled=True,
            )

            # 步态启动系数：0 -> 1
            ramp_progress = np.clip(
                gait_time / GAIT_RAMP_TIME,
                0.0,
                1.0,
            )

            # 使用Smoothstep避免启动系数突变
            ramp_alpha = (
                ramp_progress**2
                * (3.0 - 2.0 * ramp_progress)
            )

            for i, leg in enumerate(LEGS):
                q_now = q[3*i:3*i+3]

                # 完整步态生成的足端目标
                p_gait = target_dict[leg]["position"]

                # 从对称站立中心逐渐过渡到完整步态目标
                p_des = (
                    (1.0 - ramp_alpha) * centers[leg]
                    + ramp_alpha * p_gait
                )

                q_ik, error, success = inverse_kinematics(
                    leg=leg,
                    target=p_des,
                    q_init=q_now,
                    damping=0.05,
                    max_iter=100,
                    tol=1e-5,
                )

                if success:
                    q_des[3*i:3*i+3] = q_ik

        # ----------------------------------------------------
        # 执行控制
        # ----------------------------------------------------

        (
            q_control,
            dq_control,
            torque_raw,
            torque_applied,
            saturated,
        ) = apply_control(q_des)

        # 读取控制后的状态
        state = state_estimator.get_state()

        base_pos = state["position_world"]
        base_rot = state["rotation_world"]
        base_quat = data.xquat[base_id].copy()
        rpy = state["rpy"]
        velocity_world = state["velocity_world"]
        angular_velocity = state["angular_velocity_world"]

        velocity_body = base_rot.T @ velocity_world
        body_velocity_x = float(velocity_body[0])

        feet_world = get_foot_positions_world()
        feet_base = get_foot_positions_base()

        record = [
            t,
            phase,
            *base_pos,
            *base_quat,
            *rpy,
            *velocity_world,
            *angular_velocity,
            body_velocity_x,
            data.ncon,
            *q_des,
            *q_control,
            *dq_control,
            *torque_raw,
            *torque_applied,
            *saturated.astype(int),
        ]

        for leg in LEGS:
            record.extend(feet_world[leg])

        for leg in LEGS:
            record.extend(feet_base[leg])

        for leg in LEGS:
            if phase == "trot":
                record.append(
                    target_dict[leg]["foothold_offset"]
                )
            else:
                record.append(0.0)

        records.append(record)

        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed = time.time() - start_wall_time
        if data.time > elapsed:
            time.sleep(data.time - elapsed)

        if data.time >= SIM_DURATION:
            print("\n仿真时间达到设定时长")
            break


# ============================================================
# 11. 保存数据
# ============================================================

columns = [
    "time",
    "phase",
    "base_x",
    "base_y",
    "base_z",
    "base_quat_w",
    "base_quat_x",
    "base_quat_y",
    "base_quat_z",
    "roll",
    "pitch",
    "yaw",
    "base_vx_world",
    "base_vy_world",
    "base_vz_world",
    "base_wx_world",
    "base_wy_world",
    "base_wz_world",
    "base_vx_body",
    "contact_count",
]

columns += [f"{name}_des" for name in JOINT_NAMES]
columns += [f"{name}_actual" for name in JOINT_NAMES]
columns += [f"{name}_dq" for name in JOINT_NAMES]
columns += [f"{name}_tau_raw" for name in JOINT_NAMES]
columns += [f"{name}_tau" for name in JOINT_NAMES]
columns += [f"{name}_saturated" for name in JOINT_NAMES]

for leg in LEGS:
    columns += [
        f"{leg}_foot_world_x",
        f"{leg}_foot_world_y",
        f"{leg}_foot_world_z",
    ]

for leg in LEGS:
    columns += [
        f"{leg}_foot_base_x",
        f"{leg}_foot_base_y",
        f"{leg}_foot_base_z",
    ]

for leg in LEGS:
    columns.append(f"{leg}_foothold_offset")

result = pd.DataFrame(records, columns=columns)

csv_path = os.path.join(OUTPUT_DIR, "free_trot_v3_log.csv")
result.to_csv(csv_path, index=False)

print(f"\n数据已保存: {csv_path}")


# ============================================================
# 12. 统计指标
# ============================================================

base_z = result["base_z"].to_numpy()
base_x = result["base_x"].to_numpy()
base_y = result["base_y"].to_numpy()

xy_start = np.array([base_x[0], base_y[0]])
xy_end = np.array([base_x[-1], base_y[-1]])

displacement = np.linalg.norm(xy_end - xy_start)

q_des = result[
    [f"{name}_des" for name in JOINT_NAMES]
].to_numpy()

q_actual = result[
    [f"{name}_actual" for name in JOINT_NAMES]
].to_numpy()

joint_error = q_des - q_actual

saturation = result[
    [f"{name}_saturated" for name in JOINT_NAMES]
].to_numpy()

print("\n========== V3 自由机身 Trot 统计 ==========")
print(f"机身高度范围: {np.min(base_z):.4f} ~ {np.max(base_z):.4f} m")
print(f"机身水平位移: {displacement:.4f} m")
print(f"平均关节跟踪 RMSE: {np.sqrt(np.mean(joint_error**2)):.6f} rad")
print(f"最大关节跟踪误差: {np.max(np.abs(joint_error)):.6f} rad")
print(f"力矩饱和采样比例: {np.mean(saturation) * 100:.2f}%")
print(f"最大 roll: {np.rad2deg(np.max(np.abs(result['roll']))):.2f} deg")
print(f"最大 pitch: {np.rad2deg(np.max(np.abs(result['pitch']))):.2f} deg")
print(f"平均机身前向速度: {result['base_vx_body'].mean():.4f} m/s")
print("\nV3 第一轮自由机身实验结束。")