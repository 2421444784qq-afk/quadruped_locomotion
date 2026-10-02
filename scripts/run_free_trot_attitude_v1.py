import os
import time
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer

from leg_kinematics import (
    forward_kinematics,
    inverse_kinematics,
    jacobian,
)
from trot_gait import TrotGait


# ============================================================
# 1. 实验配置
# ============================================================

MODEL_PATH = (
    "unitree_mujoco/unitree_robots/go2/scene.xml"
)

OUTPUT_DIR = "outputs/free_trot_attitude_v1"
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

# 站立目标
Q_STAND_LEG = np.array([0.0, 0.8, -1.4])
Q_STAND = np.tile(Q_STAND_LEG, 4)

# PD 参数：第一轮沿用已有单腿实验参数
KP = np.tile([20.0, 20.0, 40.0], 4)
KD = np.tile([1.0, 1.0, 2.0], 4)

# 步态参数
STEP_LENGTH = 0.08
STEP_HEIGHT = 0.05
DUTY_FACTOR = 0.6
GAIT_PERIOD = 2.0

# 简化姿态反馈参数（首轮保守值，后续依据实验调整）
ATT_KP = np.array([18.0, 18.0])  # roll/pitch, Nm/rad
ATT_KD = np.array([2.0, 2.0])   # Nm/(rad/s)
ATT_TORQUE_LIMIT = 8.0          # roll/pitch合成恢复力矩限幅，Nm

# 仿真阶段
SETTLE_TIME = 2.0
TRANSITION_TIME = 1.0
HOLD_TIME = 1.0
TROT_DURATION = 3.0

TROT_START_TIME = (
    SETTLE_TIME
    + TRANSITION_TIME
    + HOLD_TIME
)
SIM_DURATION = TROT_START_TIME + TROT_DURATION


# ============================================================
# 2. 加载 MuJoCo 模型
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print("模型加载成功")
print(f"仿真时间步长: {model.opt.timestep:.6f} s")


# ============================================================
# 3. 获取关节及执行器索引
# ============================================================

qpos_ids = []
qvel_ids = []
actuator_ids = []

for joint_name, actuator_name in zip(
    JOINT_NAMES,
    ACTUATOR_NAMES
):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name
    )

    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name
    )

    if joint_id < 0 or actuator_id < 0:
        raise ValueError(
            f"未找到关节或执行器: "
            f"{joint_name}, {actuator_name}"
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
# 4. 获取机身及足端 body ID
# ============================================================

base_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base_link"
)

foot_ids = {
    leg: mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        f"{leg}_foot"
    )
    for leg in LEGS
}

if base_id < 0 or any(
    body_id < 0 for body_id in foot_ids.values()
):
    raise RuntimeError("未找到机身或足端 body")


# ============================================================
# 5. 初始化站立姿态
# ============================================================

data.qpos[qpos_ids] = Q_STAND
data.qvel[qvel_ids] = 0.0
data.ctrl[:] = 0.0

mujoco.mj_forward(model, data)

# 保存各腿站立时的足端位置，均在 base_link 坐标系下
centers = {}

for leg in LEGS:
    centers[leg] = forward_kinematics(
        leg,
        Q_STAND_LEG
    )

    print(
        f"{leg} 站立足端中心: "
        f"{centers[leg]}"
    )


# ============================================================
# 6. 创建四足 Trot 步态生成器
# ============================================================

gait = TrotGait(
    centers=centers,
    step_length=STEP_LENGTH,
    step_height=STEP_HEIGHT,
    duty_factor=DUTY_FACTOR,
    period=GAIT_PERIOD
)

print("\nTrot 步态生成器初始化完成")


# ============================================================
# 7. 状态读取
# ============================================================

def get_joint_state():
    q = data.qpos[qpos_ids].copy()
    dq = data.qvel[qvel_ids].copy()
    return q, dq


def get_base_state():
    position = data.xpos[base_id].copy()

    rotation = data.xmat[
        base_id
    ].reshape(3, 3).copy()

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

        result[leg] = (
            base_rot.T @ (foot_pos - base_pos)
        )

    return result


# ============================================================
# 8. 姿态反馈与关节控制
# ============================================================

def get_roll_pitch_and_angular_velocity():
    """返回机身roll/pitch（rad）及世界系角速度（rad/s）。"""
    rotation = data.xmat[base_id].reshape(3, 3).copy()
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2]))
    spatial_velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, base_id,
        spatial_velocity, 0
    )
    # MuJoCo spatial velocity顺序为角速度、线速度，均在世界系（flg_local=0）
    return np.array([roll, pitch]), spatial_velocity[:3].copy()


def stance_legs_at_time(gait_time):
    """按TrotGait同一相位定义，返回当前预设支撑腿。"""
    stance = []
    for leg in LEGS:
        offset = gait.phase_offsets[leg]
        cycle_phase = ((gait_time + offset * GAIT_PERIOD) % GAIT_PERIOD) / GAIT_PERIOD
        if cycle_phase < DUTY_FACTOR:
            stance.append(leg)
    return stance


def compute_attitude_compensation(gait_time):
    """姿态PD生成世界系力矩，并以支撑足点力分配后映射到关节力矩。"""
    angles, omega_world = get_roll_pitch_and_angular_velocity()
    # 小角度近似：roll/pitch反馈力矩分别沿世界X/Y轴
    moment_xy = -ATT_KP * angles - ATT_KD * omega_world[:2]
    moment_xy = np.clip(moment_xy, -ATT_TORQUE_LIMIT, ATT_TORQUE_LIMIT)
    desired_moment = np.array([moment_xy[0], moment_xy[1], 0.0])

    tau_att = np.zeros(12)
    stance = stance_legs_at_time(gait_time)
    if not stance:
        return tau_att, angles, omega_world, desired_moment, stance

    # 用最小范数解分配足端力：净力约束为0，净力矩跟踪姿态控制器输出。
    base_pos = data.xpos[base_id].copy()
    foot_world = [data.xpos[foot_ids[leg]].copy() for leg in stance]
    A = np.zeros((6, 3 * len(stance)))
    for j, foot in enumerate(foot_world):
        A[:3, 3*j:3*j+3] = np.eye(3)
        rx, ry, rz = foot - base_pos
        A[3:, 3*j:3*j+3] = np.array([
            [0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]
        ])
    wrench = np.concatenate([np.zeros(3), desired_moment])
    forces_world = np.linalg.lstsq(A, wrench, rcond=1e-5)[0].reshape(-1, 3)

    rotation = data.xmat[base_id].reshape(3, 3).copy()
    q_now, _ = get_joint_state()
    for j, leg in enumerate(stance):
        leg_i = LEGS.index(leg)
        force_base = rotation.T @ forces_world[j]
        q_leg = q_now[3*leg_i:3*leg_i+3]
        tau_att[3*leg_i:3*leg_i+3] += jacobian(leg, q_leg).T @ force_base
    return tau_att, angles, omega_world, desired_moment, stance


def apply_control(q_des, torque_attitude=None):
    q, dq = get_joint_state()

    # PD 反馈
    torque_pd = (
        KP * (q_des - q)
        - KD * dq
    )

    # 保存当前广义速度
    qvel_backup = data.qvel.copy()

    # 置零速度，提取静态偏置力
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    torque_gravity = (
        data.qfrc_bias[qvel_ids].copy()
    )

    # 恢复真实速度
    data.qvel[:] = qvel_backup
    mujoco.mj_forward(model, data)

    # 合成力矩：关节PD + 重力补偿 + 姿态反馈
    if torque_attitude is None:
        torque_attitude = np.zeros(12)
    torque_raw = torque_pd + torque_gravity + torque_attitude

    # 按模型执行器范围限幅
    torque_applied = np.clip(
        torque_raw,
        ctrl_limits[:, 0],
        ctrl_limits[:, 1]
    )

    data.ctrl[actuator_ids] = torque_applied

    saturated = (
        np.abs(torque_raw - torque_applied) > 1e-6
    )

    return (
        q,
        dq,
        torque_raw,
        torque_applied,
        saturated
    )


# ============================================================
# 9. 记录数据
# ============================================================

records = []

transition_start_positions = None
start_wall_time = time.time()

print("\n开始自由机身 Trot 仿真")
print(f"总时长: {SIM_DURATION:.2f} s")


# ============================================================
# 10. 仿真主循环
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    while viewer.is_running():

        t = data.time

        q, dq = get_joint_state()

        # 默认保持站立目标
        q_des = Q_STAND.copy()
        torque_attitude = np.zeros(12)
        attitude_angles = np.zeros(2)
        attitude_omega = np.zeros(3)
        attitude_moment = np.zeros(3)
        stance_legs = []

        # ----------------------------------------------------
        # 阶段一：自由机身站立
        # ----------------------------------------------------

        if t < SETTLE_TIME:

            phase = "stand"

        # ----------------------------------------------------
        # 阶段二：平滑过渡到 Trot 轨迹
        # ----------------------------------------------------

                # ----------------------------------------------------
        # 阶段二：平滑过渡到站立足端中心
        # ----------------------------------------------------

        elif t < SETTLE_TIME + TRANSITION_TIME:

            phase = "transition"

            alpha = (
                t - SETTLE_TIME
            ) / TRANSITION_TIME

            alpha = np.clip(alpha, 0.0, 1.0)

            # 三次平滑插值
            alpha = alpha**2 * (3.0 - 2.0 * alpha)

            # 在过渡开始时记录一次实际足端位置
            if transition_start_positions is None:
                transition_start_positions = (
                    get_foot_positions_base()
                )

            for i, leg in enumerate(LEGS):

                p_start = transition_start_positions[leg]
                p_target = centers[leg]

                # 从固定起点平滑过渡到站立中心
                p_des = (
                    (1.0 - alpha) * p_start
                    + alpha * p_target
                )

                q_now = q[3*i:3*i+3]

                q_ik, error, success = inverse_kinematics(
                    leg=leg,
                    target=p_des,
                    q_init=q_now,
                    damping=0.05,
                    max_iter=100,
                    tol=1e-5
                )

                if success:
                    q_des[3*i:3*i+3] = q_ik
               # ----------------------------------------------------
        # 阶段三：过渡后保持站立
        # ----------------------------------------------------

        elif t < TROT_START_TIME:

            phase = "hold"

            q_des = Q_STAND.copy()

        # ----------------------------------------------------
        # 阶段三：四腿周期性 Trot
        # ----------------------------------------------------

                # ----------------------------------------------------
        # 阶段四：四腿周期性 Trot
        # ----------------------------------------------------

        else:

            phase = "trot"

            gait_time = t - TROT_START_TIME

            target_positions = gait.get_all_positions(
                gait_time
            )

            for i, leg in enumerate(LEGS):

                q_now = q[3*i:3*i+3]

                q_ik, error, success = inverse_kinematics(
                    leg=leg,
                    target=target_positions[leg],
                    q_init=q_now,
                    damping=0.05,
                    max_iter=100,
                    tol=1e-5
                )

                if success:
                    q_des[3*i:3*i+3] = q_ik

        # Trot阶段启用姿态反馈；其他阶段保持原控制逻辑
        if phase == "trot":
            torque_attitude, attitude_angles, attitude_omega, attitude_moment, stance_legs = compute_attitude_compensation(gait_time)

        # ----------------------------------------------------
        # 执行控制
        # ----------------------------------------------------

        (
            q_control,
            dq_control,
            torque_raw,
            torque_applied,
            saturated
        ) = apply_control(q_des, torque_attitude)

        # 记录控制前的机身状态
        base_pos, base_rot, base_quat = get_base_state()

        base_vel = data.qvel[
            model.jnt_dofadr[
                np.where(
                    model.jnt_type
                    == mujoco.mjtJoint.mjJNT_FREE
                )[0][0]
            ]:
            model.jnt_dofadr[
                np.where(
                    model.jnt_type
                    == mujoco.mjtJoint.mjJNT_FREE
                )[0][0]
            ] + 6
        ].copy()

        feet_world = get_foot_positions_world()
        feet_base = get_foot_positions_base()

        record = [
            t,
            phase,
            *base_pos,
            *base_quat,
            *base_vel,
            data.ncon,
            *q_des,
            *q_control,
            *dq_control,
            *torque_raw,
            *torque_applied,
            *saturated.astype(int),
            *attitude_angles,
            *attitude_omega,
            *attitude_moment,
            *torque_attitude,
            len(stance_legs)
        ]

        for leg in LEGS:
            record.extend(feet_world[leg])

        for leg in LEGS:
            record.extend(feet_base[leg])

        records.append(record)

        # 推进物理仿真
        mujoco.mj_step(model, data)

        viewer.sync()

        # 与真实时间同步
        elapsed = time.time() - start_wall_time

        if data.time > elapsed:
            time.sleep(data.time - elapsed)

        if data.time >= SIM_DURATION:
            print("\n仿真时间达到设定时长")
            break


# ============================================================
# 11. 保存实验数据
# ============================================================

columns = (
    [
        "time",
        "phase",
        "base_x",
        "base_y",
        "base_z",
        "base_quat_w",
        "base_quat_x",
        "base_quat_y",
        "base_quat_z",
        "base_vx",
        "base_vy",
        "base_vz",
        "base_wx",
        "base_wy",
        "base_wz",
        "contact_count"
    ]
    + [f"{name}_des" for name in JOINT_NAMES]
    + [f"{name}_actual" for name in JOINT_NAMES]
    + [f"{name}_dq" for name in JOINT_NAMES]
    + [f"{name}_tau_raw" for name in JOINT_NAMES]
    + [f"{name}_tau" for name in JOINT_NAMES]
    + [f"{name}_saturated" for name in JOINT_NAMES]
    + ["roll", "pitch", "omega_world_x", "omega_world_y", "omega_world_z",
       "attitude_moment_x", "attitude_moment_y", "attitude_moment_z"]
    + [f"{name}_tau_attitude" for name in JOINT_NAMES]
    + ["stance_leg_count"]
)

for leg in LEGS:
    columns.extend([
        f"{leg}_foot_world_x",
        f"{leg}_foot_world_y",
        f"{leg}_foot_world_z"
    ])

for leg in LEGS:
    columns.extend([
        f"{leg}_foot_base_x",
        f"{leg}_foot_base_y",
        f"{leg}_foot_base_z"
    ])

result = pd.DataFrame(records, columns=columns)

csv_path = os.path.join(
    OUTPUT_DIR,
    "free_trot_log.csv"
)

result.to_csv(csv_path, index=False)

print(f"\n数据已保存: {csv_path}")


# ============================================================
# 12. 统计指标
# ============================================================

q_des = result[
    [f"{name}_des" for name in JOINT_NAMES]
].to_numpy()

q_actual = result[
    [f"{name}_actual" for name in JOINT_NAMES]
].to_numpy()

joint_error = q_des - q_actual

tau_raw = result[
    [f"{name}_tau_raw" for name in JOINT_NAMES]
].to_numpy()

tau = result[
    [f"{name}_tau" for name in JOINT_NAMES]
].to_numpy()

saturation = result[
    [f"{name}_saturated" for name in JOINT_NAMES]
].to_numpy()

base_z = result["base_z"].to_numpy()

print("\n========== 自由机身 Trot 统计 ==========")

print(
    f"机身高度范围: "
    f"{np.min(base_z):.4f} ~ {np.max(base_z):.4f} m"
)

# ---- 修复点：先计算位移，再做 f-string 格式化 ----
xy_start = result[["base_x", "base_y"]].iloc[0].to_numpy()
xy_end = result[["base_x", "base_y"]].iloc[-1].to_numpy()
horizontal_displacement = np.linalg.norm(xy_end - xy_start)

print(
    f"机身水平位移: "
    f"{horizontal_displacement:.4f} m"
)

print(
    f"平均关节跟踪 RMSE: "
    f"{np.sqrt(np.mean(joint_error**2)):.6f} rad"
)

print(
    f"最大关节跟踪误差: "
    f"{np.max(np.abs(joint_error)):.6f} rad"
)

print(
    f"力矩饱和采样比例: "
    f"{np.mean(saturation) * 100:.2f}%"
)

print(
    f"最大原始力矩: "
    f"{np.max(np.abs(tau_raw)):.4f} Nm"
)

print(
    f"最大输出力矩: "
    f"{np.max(np.abs(tau)):.4f} Nm"
)

print("\n姿态反馈版自由机身 Trot 实验结束。")