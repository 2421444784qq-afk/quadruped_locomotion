import numpy as np

from foot_placement import FootPlacementController


def test_compute_offset():
    controller = FootPlacementController(
        desired_velocity=0.10,
        velocity_gain=0.10,
        max_offset=0.04
    )

    # 速度低于期望值，偏移为负
    assert np.isclose(
        controller.compute_offset(0.0),
        -0.01
    )

    # 速度等于期望值，偏移为零
    assert np.isclose(
        controller.compute_offset(0.10),
        0.0
    )

    # 速度高于期望值，偏移为正
    assert np.isclose(
        controller.compute_offset(0.20),
        0.01
    )

    # 验证偏移量限幅
    assert np.isclose(
        controller.compute_offset(1.0),
        0.04
    )

    print("PASS: compute_offset")


def test_swing_transition_continuity():
    controller = FootPlacementController()

    nominal = np.array([0.2, -0.1, -0.3])

    # 先在支撑相运行
    p_support = controller.update(
        "FR", nominal, 0.2, 0.0
    )

    # 进入摆动相的起点
    p_swing_start = controller.update(
        "FR", nominal, 0.6, 0.0
    )

    # 摆动相刚开始时，偏移不应突变
    assert np.isclose(
        p_swing_start[0],
        p_support[0]
    )

    # 摆动相中间，偏移应平滑变化
    p_swing_mid = controller.update(
        "FR", nominal, 0.8, 0.0
    )

    assert p_swing_mid[0] < p_swing_start[0]

    # 摆动相接近结束时，偏移接近目标值
    p_swing_end = controller.update(
        "FR", nominal, 0.999, 0.0
    )

    expected_offset = controller.compute_offset(0.0)

    assert np.isclose(
        p_swing_end[0] - nominal[0],
        expected_offset,
        atol=1e-5
    )

    # 周期回绕后进入支撑相，偏移必须保持
    p_next_support = controller.update(
        "FR", nominal, 0.0, 0.2
    )

    assert np.isclose(
        p_next_support[0],
        p_swing_end[0],
        atol=1e-5
    )

    print("PASS: swing transition continuity")


def test_support_offset_is_held():
    controller = FootPlacementController()

    nominal = np.array([0.2, -0.1, -0.3])

    # 完成一次摆动相，建立偏移
    controller.update("FR", nominal, 0.6, 0.0)
    p_swing_end = controller.update(
        "FR", nominal, 0.999, 0.0
    )

    # 支撑相期间即使速度变化，也不应改变偏移
    p_support_1 = controller.update(
        "FR", nominal, 0.1, 0.5
    )

    p_support_2 = controller.update(
        "FR", nominal, 0.4, -0.5
    )

    assert np.isclose(
        p_support_1[0],
        p_swing_end[0],
        atol=1e-5
    )

    assert np.isclose(
        p_support_2[0],
        p_swing_end[0],
        atol=1e-5
    )

    print("PASS: support offset is held")


def test_leg_states_are_independent():
    controller = FootPlacementController()

    nominal = np.array([0.2, -0.1, -0.3])

    # FR 根据低速建立偏移
    controller.update("FR", nominal, 0.6, 0.0)
    controller.update("FR", nominal, 0.999, 0.0)

    # FL 独立运行，使用不同速度
    controller.update("FL", nominal, 0.6, 0.2)
    controller.update("FL", nominal, 0.999, 0.2)

    assert not np.isclose(
        controller.current_offsets["FR"],
        controller.current_offsets["FL"]
    )

    # 其他腿尚未运行，偏移应保持初始值
    assert np.isclose(
        controller.current_offsets["RR"],
        0.0
    )

    assert np.isclose(
        controller.current_offsets["RL"],
        0.0
    )

    print("PASS: leg states are independent")


def test_only_x_coordinate_changes():
    controller = FootPlacementController()

    nominal = np.array([0.2, -0.1, -0.3])

    corrected = controller.update(
        "FR", nominal, 0.8, 0.0
    )

    assert np.isclose(corrected[1], nominal[1])
    assert np.isclose(corrected[2], nominal[2])

    print("PASS: only X coordinate changes")


if __name__ == "__main__":
    test_compute_offset()
    test_swing_transition_continuity()
    test_support_offset_is_held()
    test_leg_states_are_independent()
    test_only_x_coordinate_changes()

    print("\nAll tests passed!")