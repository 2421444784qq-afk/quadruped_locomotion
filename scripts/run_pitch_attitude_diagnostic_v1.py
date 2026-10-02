"""Isolated pitch-feedback diagnostic for Unitree Go2 in MuJoCo.

Runs three repeatable standing trials from the same initial state:
  1) attitude compensation OFF
  2) current sign-test mapping (tau = -J.T @ F_ground)
  3) opposite mapping (tau = +J.T @ F_ground)

This is a diagnostic, not a walking controller. It uses fixed joint targets and
an initial +3 deg pitch perturbation, then logs pitch, angular velocity,
base height, actuator torques, and requested compensation moment.
"""
import os
import time
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer

from leg_kinematics import jacobian

MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"
OUTPUT_DIR = "outputs/pitch_attitude_diagnostic_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEGS = ["FR", "FL", "RR", "RL"]
JOINT_TYPES = ["hip", "thigh", "calf"]
JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in LEGS for joint in JOINT_TYPES]
ACTUATOR_NAMES = [f"{leg}_{joint}" for leg in LEGS for joint in JOINT_TYPES]
Q_STAND_LEG = np.array([0.0, 0.8, -1.4])
Q_STAND = np.tile(Q_STAND_LEG, 4)
KP = np.tile([20.0, 20.0, 40.0], 4)
KD = np.tile([1.0, 1.0, 2.0], 4)
ATT_KP_PITCH = 18.0
ATT_KD_PITCH = 2.0
ATT_LIMIT = 8.0
INITIAL_PITCH_DEG = 3.0
TRIAL_DURATION = 2.0

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

qpos_ids, qvel_ids, actuator_ids = [], [], []
for jname, aname in zip(JOINT_NAMES, ACTUATOR_NAMES):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
    if jid < 0 or aid < 0:
        raise RuntimeError(f"Missing joint/actuator: {jname}/{aname}")
    qpos_ids.append(model.jnt_qposadr[jid])
    qvel_ids.append(model.jnt_dofadr[jid])
    actuator_ids.append(aid)
qpos_ids = np.asarray(qpos_ids)
qvel_ids = np.asarray(qvel_ids)
actuator_ids = np.asarray(actuator_ids)
ctrl_limits = model.actuator_ctrlrange[actuator_ids]

base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
foot_ids = {leg: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot") for leg in LEGS}
if base_id < 0 or any(v < 0 for v in foot_ids.values()):
    raise RuntimeError("Could not find base_link or foot bodies")

# Find the free joint attached to the base body.
root_joint_id = None
for jid in range(model.njnt):
    if model.jnt_bodyid[jid] == base_id and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
        root_joint_id = jid
        break
if root_joint_id is None:
    raise RuntimeError("No free joint found on base_link")
root_qadr = model.jnt_qposadr[root_joint_id]


def get_roll_pitch_omega():
    R = data.xmat[base_id].reshape(3, 3)
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))
    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, vel6, 0)
    return roll, pitch, vel6[:3].copy()


def set_trial_initial_state():
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[qpos_ids] = Q_STAND
    data.qvel[qvel_ids] = 0.0
    # Free-joint qpos: xyz followed by quaternion (w, x, y, z).
    half = np.deg2rad(INITIAL_PITCH_DEG) / 2.0
    data.qpos[root_qadr + 3:root_qadr + 7] = [np.cos(half), 0.0, np.sin(half), 0.0]
    data.qvel[root_qadr:root_qadr + 6] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def calculate_pitch_compensation(sign):
    """Allocate a pure world-Y moment over four assumed stance feet."""
    roll, pitch, omega = get_roll_pitch_omega()
    desired_my = float(np.clip(-ATT_KP_PITCH * pitch - ATT_KD_PITCH * omega[1], -ATT_LIMIT, ATT_LIMIT))
    desired_wrench = np.array([0.0, 0.0, 0.0, 0.0, desired_my, 0.0])

    # Use the subtree COM as moment reference for this diagnostic.
    com = data.subtree_com[base_id].copy()
    A = np.zeros((6, 12))
    feet_world = [data.xpos[foot_ids[leg]].copy() for leg in LEGS]
    for i, p in enumerate(feet_world):
        rx, ry, rz = p - com
        A[:3, 3*i:3*i+3] = np.eye(3)
        A[3:, 3*i:3*i+3] = np.array([
            [0.0, -rz, ry],
            [rz, 0.0, -rx],
            [-ry, rx, 0.0],
        ])
    forces_world = np.linalg.lstsq(A, desired_wrench, rcond=1e-6)[0].reshape(4, 3)

    R = data.xmat[base_id].reshape(3, 3).copy()
    q = data.qpos[qpos_ids].copy()
    tau = np.zeros(12)
    for i, leg in enumerate(LEGS):
        q_leg = q[3*i:3*i+3]
        force_base = R.T @ forces_world[i]
        tau[3*i:3*i+3] = sign * (jacobian(leg, q_leg).T @ force_base)
    return tau, desired_my, forces_world


def apply_pd_and_gravity(tau_att):
    q = data.qpos[qpos_ids].copy()
    dq = data.qvel[qvel_ids].copy()
    tau_pd = KP * (Q_STAND - q) - KD * dq
    qvel_backup = data.qvel.copy()
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    tau_g = data.qfrc_bias[qvel_ids].copy()
    data.qvel[:] = qvel_backup
    mujoco.mj_forward(model, data)
    raw = tau_pd + tau_g + tau_att
    applied = np.clip(raw, ctrl_limits[:, 0], ctrl_limits[:, 1])
    data.ctrl[actuator_ids] = applied
    return raw, applied


TRIALS = [
    ("no_feedback", 0),
    ("negative_JT", -1),
    ("positive_JT", +1),
]
records = []
print("Pitch 姿态反馈独立诊断启动")
print(f"初始扰动: +{INITIAL_PITCH_DEG:.1f} deg；每组时长: {TRIAL_DURATION:.1f} s")

with mujoco.viewer.launch_passive(model, data) as viewer:
    for trial_name, sign in TRIALS:
        set_trial_initial_state()
        print(f"\n开始试验: {trial_name}")
        n_steps = int(TRIAL_DURATION / model.opt.timestep)
        for k in range(n_steps):
            t = data.time
            if sign == 0:
                tau_att = np.zeros(12)
                desired_my = 0.0
                forces_world = np.zeros((4, 3))
            else:
                tau_att, desired_my, forces_world = calculate_pitch_compensation(sign)
            tau_raw, tau_applied = apply_pd_and_gravity(tau_att)
            roll, pitch, omega = get_roll_pitch_omega()
            records.append({
                "trial": trial_name,
                "time": t,
                "roll_rad": roll,
                "pitch_rad": pitch,
                "roll_deg": np.rad2deg(roll),
                "pitch_deg": np.rad2deg(pitch),
                "omega_x": omega[0],
                "omega_y": omega[1],
                "omega_z": omega[2],
                "base_z": data.xpos[base_id][2],
                "desired_pitch_moment_y": desired_my,
                "tau_attitude_max_abs": np.max(np.abs(tau_att)),
                "tau_raw_max_abs": np.max(np.abs(tau_raw)),
                "tau_applied_max_abs": np.max(np.abs(tau_applied)),
                "contact_count": int(data.ncon),
                "com_x": data.subtree_com[base_id][0],
                "com_y": data.subtree_com[base_id][1],
                "com_z": data.subtree_com[base_id][2],
                **{f"{leg}_tau_att_{j}": tau_att[3*i+j] for i, leg in enumerate(LEGS) for j in range(3)},
                **{f"{leg}_force_{axis}": forces_world[i, a] for i, leg in enumerate(LEGS) for a, axis in enumerate("xyz")},
            })
            mujoco.mj_step(model, data)
            viewer.sync()
        # Briefly pause between trials so the user can observe reset state.
        time.sleep(0.5)

csv_path = os.path.join(OUTPUT_DIR, "pitch_diagnostic_log.csv")
pd.DataFrame(records).to_csv(csv_path, index=False)
print(f"\n数据已保存: {csv_path}")
for name, _ in TRIALS:
    df = pd.DataFrame([r for r in records if r["trial"] == name])
    first = df.iloc[0]
    last = df.iloc[-1]
    print(
        f"{name}: pitch {first['pitch_deg']:.2f}° -> {last['pitch_deg']:.2f}°, "
        f"max|pitch|={df['pitch_deg'].abs().max():.2f}°, "
        f"base_z min={df['base_z'].min():.3f} m"
    )
print("诊断完成。注意：这是站立小扰动试验，不能直接等同于动态 Trot 稳定性。")
