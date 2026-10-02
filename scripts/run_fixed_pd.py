
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mujoco


# ============================================================
# 1. 路径与实验参数
# ============================================================

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene_fixed.xml"
TRAJ_PATH = "outputs/trot/trot_joint_trajectory.csv"
OUTPUT_DIR = "outputs/fixed_pd"

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

# 第一轮使用统一 PD 增益，后续再根据响应调整
KP = np.array([40.0, 40.0, 40.0] * 4)
KD = np.array([2.0, 2.0, 2.0] * 4)


# ============================================================
# 2. 加载模型与参考轨迹
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

df = pd.read_csv(TRAJ_PATH)
t_ref = df["time"].to_numpy()

q_ref = df[JOINT_NAMES].to_numpy()

# 由参考位置轨迹数值计算参考速度
qd_ref = np.gradient(q_ref, t_ref, axis=0)

# 执行器力矩上限
ctrl_limits = model.actuator_ctrlrange.copy()

# 通过关节名称获取 qpos 和 qvel 地址
qpos_ids = []
qvel_ids = []
actuator_ids = []

for joint_name, actuator_name in zip(
    JOINT_NAMES, ACTUATOR_NAMES
):
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

print("模型加载成功")
print(f"仿真时间步长: {model.opt.timestep:.6f} s")
print(f"参考轨迹时长: {t_ref[-1]:.3f} s")
print(f"参考采样间隔: {np.mean(np.diff(t_ref)):.4f} s")


# ============================================================
# 3. 初始化
# ============================================================

# 以参考轨迹第0帧作为初始姿态
data.qpos[qpos_ids] = q_ref[0]
data.qvel[qvel_ids] = 0.0

data.ctrl[:] = 0.0

mujoco.mj_forward(model, data)


# ============================================================
# 4. PD 闭环仿真
# ============================================================

sim_dt = model.opt.timestep
sim_duration = t_ref[-1]

records = []

while data.time < sim_duration:

    t = data.time

    # 将参考轨迹插值到 MuJoCo 仿真时间步
    q_des = np.array([
        np.interp(t, t_ref, q_ref[:, i])
        for i in range(12)
    ])

    qd_des = np.array([
        np.interp(t, t_ref, qd_ref[:, i])
        for i in range(12)
    ])

    q = data.qpos[qpos_ids].copy()
    qd = data.qvel[qvel_ids].copy()

    # PD 控制律
    error = q_des - q
    error_d = qd_des - qd

    tau_raw = KP * error + KD * error_d

    # 根据执行器限幅进行饱和处理
    tau = np.clip(
        tau_raw,
        ctrl_limits[actuator_ids, 0],
        ctrl_limits[actuator_ids, 1]
    )

    data.ctrl[actuator_ids] = tau

    # 记录控制前的状态
    records.append(
        np.concatenate([
            [t],
            q_des,
            q,
            qd_des,
            qd,
            tau,
            tau_raw
        ])
    )

    mujoco.mj_step(model, data)


# ============================================================
# 5. 整理仿真数据
# ============================================================

records = np.array(records)

columns = (
    ["time"]
    + [f"{n}_des" for n in JOINT_NAMES]
    + [f"{n}_actual" for n in JOINT_NAMES]
    + [f"{n}_qd_des" for n in JOINT_NAMES]
    + [f"{n}_qd_actual" for n in JOINT_NAMES]
    + [f"{n}_tau" for n in JOINT_NAMES]
    + [f"{n}_tau_raw" for n in JOINT_NAMES]
)

result = pd.DataFrame(records, columns=columns)

result.to_csv(
    os.path.join(OUTPUT_DIR, "pd_tracking.csv"),
    index=False
)


# ============================================================
# 6. 计算跟踪误差指标
# ============================================================

q_actual = result[
    [f"{n}_actual" for n in JOINT_NAMES]
].to_numpy()

q_desired = result[
    [f"{n}_des" for n in JOINT_NAMES]
].to_numpy()

error = q_desired - q_actual

rmse = np.sqrt(np.mean(error ** 2, axis=0))
mae = np.mean(np.abs(error), axis=0)
max_error = np.max(np.abs(error), axis=0)

tau = result[
    [f"{n}_tau" for n in JOINT_NAMES]
].to_numpy()

tau_raw = result[
    [f"{n}_tau_raw" for n in JOINT_NAMES]
].to_numpy()

saturation = np.abs(tau_raw - tau) > 1e-6
saturation_rate = np.mean(saturation, axis=0) * 100

metrics = pd.DataFrame({
    "joint": JOINT_NAMES,
    "RMSE_rad": rmse,
    "MAE_rad": mae,
    "MaxError_rad": max_error,
    "SaturationRate_percent": saturation_rate
})

metrics.to_csv(
    os.path.join(OUTPUT_DIR, "pd_metrics.csv"),
    index=False
)

print("\n===== 关节跟踪指标 =====")
print(metrics.to_string(index=False))

print("\n===== 整体指标 =====")
print(f"平均关节 RMSE: {np.mean(rmse):.6f} rad")
print(f"平均关节 MAE: {np.mean(mae):.6f} rad")
print(f"最大关节误差: {np.max(max_error):.6f} rad")
print(f"平均力矩饱和率: {np.mean(saturation_rate):.2f}%")


# ============================================================
# 7. 绘制关节跟踪曲线
# ============================================================

fig, axes = plt.subplots(
    4, 3, figsize=(15, 12), sharex=True
)

time = result["time"].to_numpy()

for i, ax in enumerate(axes.flat):
    ax.plot(
        time, q_desired[:, i],
        label="Reference", linewidth=1.5
    )
    ax.plot(
        time, q_actual[:, i],
        "--", label="Actual", linewidth=1.2
    )

    ax.set_title(JOINT_NAMES[i])
    ax.set_ylabel("Angle (rad)")
    ax.grid(True, alpha=0.3)

    if i == 0:
        ax.legend()

for ax in axes[-1, :]:
    ax.set_xlabel("Time (s)")

fig.suptitle(
    "Fixed-base Go2 Joint PD Tracking",
    fontsize=15
)
fig.tight_layout()

fig.savefig(
    os.path.join(OUTPUT_DIR, "joint_tracking.png"),
    dpi=200
)

plt.close(fig)


# ============================================================
# 8. 绘制关节误差曲线
# ============================================================

fig, axes = plt.subplots(
    4, 3, figsize=(15, 12), sharex=True
)

for i, ax in enumerate(axes.flat):
    ax.plot(time, error[:, i], linewidth=1.2)
    ax.set_title(JOINT_NAMES[i])
    ax.set_ylabel("Error (rad)")
    ax.grid(True, alpha=0.3)

for ax in axes[-1, :]:
    ax.set_xlabel("Time (s)")

fig.suptitle(
    "Joint Tracking Error",
    fontsize=15
)
fig.tight_layout()

fig.savefig(
    os.path.join(OUTPUT_DIR, "joint_error.png"),
    dpi=200
)

plt.close(fig)

print(f"\n实验完成，结果保存在: {OUTPUT_DIR}")
