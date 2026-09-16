import numpy as np

FULL_SCALE_DUTY = 1000.0
CURRENT_AMPS_PER_TICK = 0.0065


def encoder_tick_radians(ticks_per_revolution: int) -> float:
    return 2.0 * np.pi / ticks_per_revolution


def commanded_duty(
    position_error_radians: np.ndarray,
    ticks_per_revolution: int,
    duty_per_tick: float,
    dead_zone_ticks: int,
    min_startup_duty: float,
    max_duty: np.ndarray,
) -> np.ndarray:
    """
    What an STS3215 does with a position error: round to whole encoder ticks, scale by the firmware
    gain, ignore anything inside the dead zone or too weak to break stiction, and clip at the duty
    limit. The result is in the same units the servo reports as Present_Load, 0 to 1000.

    MuJoCo's position actuator does none of this, which is why sim reacts to errors the real servo
    discards and saturates two orders of magnitude sooner.
    """
    error_ticks = np.round(position_error_radians / encoder_tick_radians(ticks_per_revolution))
    error_ticks = np.where(np.abs(error_ticks) <= dead_zone_ticks, 0.0, error_ticks)

    duty = np.clip(duty_per_tick * error_ticks, -max_duty, max_duty)
    return np.where(np.abs(duty) < min_startup_duty, 0.0, duty)


def duty_from_action(
    action: np.ndarray,
    full_scale_duty: float,
    min_startup_duty: float,
    max_duty: np.ndarray,
) -> np.ndarray:
    """
    What the servo does with a duty commanded directly in its open loop PWM mode. There is no
    position error, so the encoder dead zone does not apply, but the duty limit and the duty too
    weak to break stiction still do.
    """
    duty = np.clip(action * full_scale_duty, -max_duty, max_duty)
    return np.where(np.abs(duty) < min_startup_duty, 0.0, duty)


def duty_to_torque(
    duty: np.ndarray,
    joint_velocity_radians_per_second: np.ndarray,
    stall_torque_newton_meters: float,
    no_load_speed_radians_per_second: float,
) -> np.ndarray:
    """
    A duty sets voltage across the motor, not force. The motor generates an opposing voltage in
    proportion to its own speed, so the same duty yields full torque when blocked and none at the
    top speed that duty can reach.
    """
    return stall_torque_newton_meters * (duty / FULL_SCALE_DUTY - joint_velocity_radians_per_second / no_load_speed_radians_per_second)
