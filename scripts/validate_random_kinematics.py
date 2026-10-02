
import mujoco
import numpy as np

from leg_kinematics import forward_kinematics, jacobian


MODEL_PATH = "unitree_mujoco/unitree_robots/go2/scene.xml"

LEGS = {
    "FR": {
        "joints": ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
        "foot": "FR_foot",
        "limits": [
            [-1.0472, 1.0472],
            [-1.5708, 3.4907],
            [-2.7227, -0.83776],
        ],
    },
    "FL": {
        "joints": ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
        "foot": "FL_foot",
        "limits": [
            [-1.0472, 1.0472],
            [-1.5708, 3.4907],
            [-2.7227, -0.83776],
        ],
    },
    "RR": {
        "joints": ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
        "foot": "RR_foot",
        "limits": [
            [-1.0472, 1.0472],
            [-0.5236, 4.5379],
            [-2.7227, -0.83776],
        ],
    },
    "RL": {
        "joints": ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
        "foot": "RL_foot",
        "limits": [
            [-1.0472, 1.0472],
            [-0.5236, 4.5379],
            [-2.7227, -0.83776],
        ],
    },
}

NUM_SAMPLES = 100
EPS = 1e-6
RNG = np.random.default_rng(42)


def get_body_position(model, data, body_name):
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, body_name
    )
    if body_id < 0:
        raise ValueError(f"找不到 body: {body_name}")
    return data.xpos[body_id].copy()


def get_joint_qpos_addresses(model, joint_names):
    addresses = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise ValueError(f"找不到关节: {name}")
        addresses.append(model.jnt_qposadr[joint_id])
    return addresses


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    base_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
    )
    if base_id < 0:
        raise ValueError("找不到 base_link")

    print("========== 随机姿态运动学验证 ==========")
    print(f"每条腿测试姿态数: {NUM_SAMPLES}")
    print(f"总测试姿态数: {NUM_SAMPLES * len(LEGS)}")

    overall_fk_max = 0.0
    overall_jac_max = 0.0
    all_passed = True

    for leg_name, cfg in LEGS.items():
        qpos_addresses = get_joint_qpos_addresses(
            model, cfg["joints"]
        )

        foot_fk_max = 0.0
        jac_max = 0.0

        for sample_idx in range(NUM_SAMPLES):
            # 在关节限位范围内随机采样
            limits = np.array(cfg["limits"])
            q = RNG.uniform(limits[:, 0], limits[:, 1])

            # 重置到 Go2 默认关键帧
            mujoco.mj_resetDataKeyframe(model, data, 0)

            # 设置当前腿的关节角
            for address, angle in zip(qpos_addresses, q):
                data.qpos[address] = angle

            mujoco.mj_forward(model, data)

            # MuJoCo 足端相对机身的位置
            base_pos = get_body_position(
                model, data, "base_link"
            )
            foot_pos = get_body_position(
                model, data, cfg["foot"]
            )
            target = foot_pos - base_pos

            # 解析 FK
            predicted = forward_kinematics(leg_name, q)
            fk_error = np.linalg.norm(predicted - target)
            foot_fk_max = max(foot_fk_max, fk_error)

            # 解析 Jacobian
            J_analytic = jacobian(leg_name, q)

            # 数值差分 Jacobian
            J_numeric = np.zeros((3, 3))

            for j in range(3):
                q_plus = q.copy()
                q_minus = q.copy()

                q_plus[j] += EPS
                q_minus[j] -= EPS

                mujoco.mj_resetDataKeyframe(model, data, 0)
                for address, angle in zip(qpos_addresses, q_plus):
                    data.qpos[address] = angle
                mujoco.mj_forward(model, data)

                pos_plus = (
                    get_body_position(model, data, cfg["foot"])
                    - get_body_position(model, data, "base_link")
                )

                mujoco.mj_resetDataKeyframe(model, data, 0)
                for address, angle in zip(qpos_addresses, q_minus):
                    data.qpos[address] = angle
                mujoco.mj_forward(model, data)

                pos_minus = (
                    get_body_position(model, data, cfg["foot"])
                    - get_body_position(model, data, "base_link")
                )

                J_numeric[:, j] = (
                    pos_plus - pos_minus
                ) / (2.0 * EPS)

            jac_error = np.max(
                np.abs(J_analytic - J_numeric)
            )
            jac_max = max(jac_max, jac_error)

        overall_fk_max = max(overall_fk_max, foot_fk_max)
        overall_jac_max = max(overall_jac_max, jac_max)

        passed = foot_fk_max < 1e-8 and jac_max < 1e-5
        all_passed = all_passed and passed

        print(f"\n{leg_name}:")
        print(f"  FK 最大误差: {foot_fk_max:.10e} m")
        print(f"  Jacobian 最大误差: {jac_max:.10e}")
        print(f"  结果: {'PASS' if passed else 'FAIL'}")

    print("\n========== 总结 ==========")
    print(f"全部姿态 FK 最大误差: {overall_fk_max:.10e} m")
    print(f"全部姿态 Jacobian 最大误差: {overall_jac_max:.10e}")
    print(f"最终结果: {'全部通过' if all_passed else '存在未通过项'}")


if __name__ == "__main__":
    main()
