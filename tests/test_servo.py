import numpy as np

from robot_arm.backends.servo import commanded_duty, compensated_duty, duty_to_torque, encoder_tick_radians

# The values read off the arm in outputs/characterize_gripper/2026-08-31/17-47-08.
TICKS_PER_REVOLUTION = 4096
DUTY_PER_TICK = 4.6
DEAD_ZONE_TICKS = 1
MIN_STARTUP_FORCE = 16
MAX_TORQUE_LIMIT = 1000


def duty_for(error_ticks: float, max_torque_limit: float = MAX_TORQUE_LIMIT) -> float:
    error_radians = error_ticks * encoder_tick_radians(TICKS_PER_REVOLUTION)
    return float(
        commanded_duty(
            np.array([error_radians]),
            TICKS_PER_REVOLUTION,
            DUTY_PER_TICK,
            DEAD_ZONE_TICKS,
            MIN_STARTUP_FORCE,
            np.array([max_torque_limit]),
        )[0]
    )


def test_duty_reproduces_the_measured_stall_point():
    # The gripper held 40 ticks of error against an object and reported 0.184 of full scale.
    np.testing.assert_allclose(duty_for(40), 184.0)


def test_small_errors_fall_below_the_startup_threshold():
    # 3 ticks is 13.8 duty, under Min_Startup_Force, which is why 4 ticks moved nothing on the arm.
    assert duty_for(3) == 0.0
    assert duty_for(4) > 0.0


def test_sub_tick_errors_round_away():
    assert duty_for(0.4) == 0.0


def test_duty_clips_at_the_per_motor_torque_limit():
    # lerobot writes 500 for the gripper and leaves 1000 everywhere else.
    assert duty_for(1000, max_torque_limit=500) == 500.0
    assert duty_for(1000, max_torque_limit=1000) == 1000.0


def test_compensated_duty_centers_zero_on_compensation_and_preserves_limits():
    compensation = np.array([0.3, -0.2, 0.1])
    limits = np.array([1.0, 1.0, 0.5])

    np.testing.assert_allclose(compensated_duty(np.zeros(3), compensation, limits), compensation)
    np.testing.assert_allclose(compensated_duty(np.ones(3), compensation, limits), limits)
    np.testing.assert_allclose(compensated_duty(-np.ones(3), compensation, limits), -limits)


def test_compensated_duty_scales_each_side_over_its_available_range():
    compensation = np.array([0.3, 0.3])
    limits = np.ones(2)
    policy_action = np.array([0.5, -0.5])

    np.testing.assert_allclose(compensated_duty(policy_action, compensation, limits), [0.65, -0.35])


def test_torque_is_full_at_stall_and_zero_at_no_load_speed():
    stall_torque = 1.31
    no_load_speed = 3.35

    np.testing.assert_allclose(duty_to_torque(np.array([1000.0]), np.array([0.0]), stall_torque, no_load_speed), [stall_torque])
    np.testing.assert_allclose(duty_to_torque(np.array([1000.0]), np.array([no_load_speed]), stall_torque, no_load_speed), [0.0], atol=1e-12)


def test_torque_brakes_when_moving_faster_than_the_duty_commands():
    stall_torque = 1.31
    no_load_speed = 3.35

    torque = duty_to_torque(np.array([0.0]), np.array([1.0]), stall_torque, no_load_speed)
    assert torque[0] < 0.0
