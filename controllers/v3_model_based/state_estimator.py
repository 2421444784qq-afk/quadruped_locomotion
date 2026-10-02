import numpy as np
import mujoco


class StateEstimator:
    """
    Go2 机身状态读取模块。

    输出：
        position_world: 机身世界坐标位置
        rotation_world: 机身坐标系到世界坐标系的旋转矩阵
        rpy: roll, pitch, yaw
        velocity_world: 机身线速度（世界坐标系）
        angular_velocity_world: 机身角速度（世界坐标系）
    """

    def __init__(self, model, data, body_name="base"):
        self.model = model
        self.data = data

        self.body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )

        if self.body_id < 0:
            raise ValueError(
                f"Body '{body_name}' not found. "
                "Please check the body name in the XML."
            )

    def get_state(self):
        data = self.data
        bid = self.body_id

        position = data.xpos[bid].copy()
        rotation = data.xmat[bid].reshape(3, 3).copy()

        # MuJoCo cvel 排列为 [角速度(3), 线速度(3)]，是 c-frame：
        # 以该 body 的子树质心(subtree_com)为中心、坐标轴与世界系同向。
        # 因此角速度本身已是世界系；线速度是子树质心的世界系速度。
        spatial_velocity = data.cvel[bid].copy()

        angular_world = spatial_velocity[:3]
        linear_com_world = spatial_velocity[3:]

        # 把线速度从子树质心传递到 base 坐标系原点：
        #   v_origin = v_com - ω × (com - origin)
        com = data.subtree_com[bid].copy()
        linear_world = linear_com_world - np.cross(
            angular_world, com - position
        )

        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arcsin(
            np.clip(-rotation[2, 0], -1.0, 1.0)
        )
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])

        return {
            "position_world": position,
            "rotation_world": rotation,
            "rpy": np.array([roll, pitch, yaw]),
            "velocity_world": linear_world,
            "angular_velocity_world": angular_world,
        }
