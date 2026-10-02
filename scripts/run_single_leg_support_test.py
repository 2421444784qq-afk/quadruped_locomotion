
import os
import time
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer

from leg_kinematics import forward_kinematics, inverse_kinematics


# ============================================================
# 1. 实验配置
# ============================================================

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
OUTPUT_DIR = "outputs/single_leg_support_test"
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

# 沿用V3中已经测试过的站立姿态
Q_STAND_LEG = np.array([0.0, 0.8, -1.45])
Q_STAND = np.tile(Q_STAND_LEG, 4)

KP = np.tile([20.0, 20.0, 40.0], 4)
KD = np.tile([1.0, 1.0, 2.0], 4)

# 单腿抬升高度：2 cm
LIFT_HEIGHT = 0.02

# 每条腿的实验阶段时长
SETTLE_TIME = 2.0
LIFT_TIME = 1.0
HOLD_TIME = 1.0
LOWER_TIME = 1.0
RECOVERY_TIME = 1.0

TEST_TIME = (
    SETTLE_TIME
    + LIFT_TIME
    + HOLD_TIME
    + LOWER_TIME
    + RECOVERY_TIME
)

TOTAL_DURATION = TEST_TIME * len(LEGS)

# 安全阈值：出现明显倾倒趋势时停止
MIN_BASE_HEIGHT = 0.18
MAX_TILT_DEG = 25.0

# 仿真记录频率
LOG_INTERVAL = 10


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


# ============================================================
# 4. 获取机身与足端ID
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

if base_id < 0 or any(x < 0 for x in foot_ids.values()):
    raise RuntimeError("未找到机身或足端body")


# ============================================================
# 5. 初始化站立姿态
# ============================================================

data.qpos[qpos_ids] = Q_STAND
data.qvel[qvel_ids] = 0.0
data.ctrl[:] = 0.0

mujoco.mj_forward(model, data)

centers = {
    leg: forward_kinematics(leg, Q_STAND_LEG)
    for leg in LEGS
}

print("\n站立足端中心：")
for leg in LEGS:
    print(f"{leg}: {centers[leg]}")


# ============================================================
# 6. 状态读取
# ============================================================

def get_joint_state():
    return (
        data.qpos[qpos_ids].copy(),
        data.qvel[qvel_ids].copy(),
    )


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


# ============================================================
# 7. PD + 重力补偿
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
# 8. 平滑插值
# ============================================================

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ============================================================
# 9. 实验主循环
# ============================================================

records = []
step_count = 0
aborted = False
abort_reason = ""

print("\n开始单腿支撑稳定性实验")
print(f"测试顺序: {LEGS}")
print(f"单腿抬升高度: {LIFT_HEIGHT:.3f} m")
print(f"计划总时长: {TOTAL_DURATION:.2f} s")

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running() and data.time < TOTAL_DURATION:

        t = data.time

        # 当前测试哪条腿
        test_index = min(
            int(t // TEST_TIME),
            len(LEGS) - 1,
        )

        active_leg = LEGS[test_index]
        local_time = t - test_index * TEST_TIME

        q, dq = get_joint_state()
        q_des = Q_STAND.copy()

        phase = "settle"
        lift_alpha = 0.0

        # ----------------------------------------------------
        # 阶段一：静态站立
        # ----------------------------------------------------

        if local_time < SETTLE_TIME:
            phase = "settle"

        # ----------------------------------------------------
        # 阶段二：缓慢抬腿
        # ----------------------------------------------------

        elif local_time < SETTLE_TIME + LIFT_TIME:

            phase = "lift"

            progress = (
                local_time - SETTLE_TIME
            ) / LIFT_TIME

            lift_alpha = smoothstep(progress)

        # ----------------------------------------------------
        # 阶段三：保持抬腿
        # ----------------------------------------------------

        elif local_time < SETTLE_TIME + LIFT_TIME + HOLD_TIME:

            phase = "hold"
            lift_alpha = 1.0

        # ----------------------------------------------------
        # 阶段四：缓慢放腿
        # ----------------------------------------------------

        elif local_time < (
            SETTLE_TIME + LIFT_TIME + HOLD_TIME + LOWER_TIME
        ):

            phase = "lower"

            progress = (
                local_time
                - SETTLE_TIME
                - LIFT_TIME
                - HOLD_TIME
            ) / LOWER_TIME

            lift_alpha = 1.0 - smoothstep(progress)

        # ----------------------------------------------------
        # 阶段五：恢复站立
        # ----------------------------------------------------

        else:
            phase = "recovery"
            lift_alpha = 0.0

        # ----------------------------------------------------
        # 仅修改当前测试腿的足端目标
        # ----------------------------------------------------

        i = LEGS.index(active_leg)
        q_now = q[3*i:3*i+3]

        p_target = centers[active_leg].copy()
        p_target[2] += LIFT_HEIGHT * lift_alpha

        q_ik, error, success = inverse_kinematics(
            leg=active_leg,
            target=p_target,
            q_init=q_now,
            damping=0.05,
            max_iter=100,
            tol=1e-5,
        )

        if success:
            q_des[3*i:3*i+3] = q_ik

        # 其余三条腿保持站立关节目标
        (
            q_control,
            dq_control,
            torque_raw,
            torque_applied,
            saturated,
        ) = apply_control(q_des)

        # ----------------------------------------------------
        # 读取机身状态
        # ----------------------------------------------------

        base_pos, base_rot, base_quat = get_base_state()

        # 欧拉角：roll, pitch, yaw
        pitch = np.arcsin(
            np.clip(-base_rot[2, 0], -1.0, 1.0)
        )

        roll = np.arctan2(
            base_rot[2, 1],
            base_rot[2, 2],
        )

        yaw = np.arctan2(
            base_rot[1, 0],
            base_rot[0, 0],
        )

        feet_world = get_foot_positions_world()

        # ----------------------------------------------------
        # 记录数据
        # ----------------------------------------------------

        if step_count % LOG_INTERVAL == 0:

            record = [
                t,
                active_leg,
                phase,
                lift_alpha,
                *base_pos,
                *base_quat,
                roll,
                pitch,
                yaw,
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

            records.append(record)

        step_count += 1

        # ----------------------------------------------------
        # 安全保护
        # ----------------------------------------------------

        tilt_deg = np.rad2deg(
            max(abs(roll), abs(pitch))
        )

        if base_pos[2] < MIN_BASE_HEIGHT:
            aborted = True
            abort_reason = (
                f"机身高度低于安全阈值: {base_pos[2]:.3f} m"
            )

        elif tilt_deg > MAX_TILT_DEG:
            aborted = True
            abort_reason = (
                f"机身倾角超过安全阈值: {tilt_deg:.2f} deg"
            )

        if aborted:
            print(f"\n触发安全停止：{abort_reason}")
            break

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(max(0.0, model.opt.timestep))

# ============================================================
# 10. 保存CSV
# ============================================================

columns = [
    "time",
    "active_leg",
    "phase",
    "lift_alpha",
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

result = pd.DataFrame(records, columns=columns)

csv_path = os.path.join(
    OUTPUT_DIR,
    "single_leg_support_log.csv",
)

result.to_csv(csv_path, index=False)

print(f"\n数据已保存: {csv_path}")

if aborted:
    print(f"实验提前停止: {abort_reason}")
else:
    print("所有单腿测试阶段完成。")

print("\n单腿支撑稳定性实验结束。")
