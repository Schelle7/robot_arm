from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from robot_arm.robot_schema import MOTOR_ORDER, POLICY_OBSERVATION_NAMES


COMPATIBILITY_FIELDS = (
    "control.frequencies.cartesian",
    "control.frequencies.joint",
    "control.policy_history_seconds",
    "control.joint_velocity_scale_radians_per_second",
    "waypoint.cartesian_action_dim",
    "waypoint.position_speed_meters_per_second",
    "waypoint.rotation_speed_radians_per_second",
    "waypoint.gripper_speed_radians_per_second",
    "training.terminate_at_cartesian_action_end",
    "servo.full_scale_duty",
    "servo.max_duty",
    "safety.duty_ema_seconds",
    "safety.max_smoothed_duty",
    "reward",
)


def _config_value(cfg: DictConfig, field: str):
    value = cfg
    for key in field.split("."):
        value = value[key]
    return value


def validate_real_recording_config(recorded_cfg: DictConfig, training_cfg: DictConfig) -> None:
    assert recorded_cfg.backend == "real", "Real replay data must come from a real backend"
    assert recorded_cfg.runtime.record_policy_debug, "Real recordings must include joint debug transitions"
    for field in COMPATIBILITY_FIELDS:
        recorded = _config_value(recorded_cfg, field)
        current = _config_value(training_cfg, field)
        if recorded != current:
            raise ValueError(f"Real recording configuration mismatch for {field}: recorded {recorded}, training {current}")


def _validate_transition(step: dict, buffer, duty_limits: np.ndarray) -> None:
    for field, storage in (("obs", buffer.observations), ("next_obs", buffer.next_observations)):
        assert set(step[field]) == set(POLICY_OBSERVATION_NAMES), f"Unexpected {field} fields"
        for name in POLICY_OBSERVATION_NAMES:
            values = np.asarray(step[field][name])
            assert values.shape == storage[name].shape[1:], f"Invalid {field}.{name} shape: {values.shape}"
            assert np.all(np.isfinite(values)), f"Non-finite {field}.{name}"
    action = np.asarray(step["action"])
    assert action.shape == buffer.actions.shape[1:], f"Invalid action shape: {action.shape}"
    assert np.all(np.isfinite(action)) and np.all(np.abs(action) <= 1.0), "Invalid normalized policy action"
    requested_duty = np.asarray(step["requested_duty"])
    assert requested_duty.shape == action.shape
    assert np.allclose(requested_duty, action * duty_limits), "Recorded duty does not match direct-duty scaling"
    assert np.ndim(step["reward"]) == 0 and np.isfinite(step["reward"]), "Invalid reward"
    assert isinstance(step["terminated"], (bool, np.bool_)), "Termination flag must be boolean"


def load_real_transitions(episode_paths: list[str], cfg: DictConfig, buffer) -> int:
    paths = [Path(path).expanduser().resolve() for path in episode_paths]
    assert len(set(paths)) == len(paths), "Duplicate real episode paths"
    duty_limits = np.array([cfg.servo.max_duty[name] / cfg.servo.full_scale_duty for name in MOTOR_ORDER])
    loaded = 0
    for path in paths:
        recorded_cfg = OmegaConf.load(path.parents[2] / ".hydra" / "config.yaml")
        validate_real_recording_config(recorded_cfg, cfg)
        with np.load(path, allow_pickle=True) as recording:
            steps = [step for trajectory in recording["dense_trajectory"] for step in trajectory]
        assert len(steps) > 0, f"No joint transitions in {path}"
        if buffer.size + len(steps) > buffer.capacity:
            raise ValueError(f"Real replay capacity {buffer.capacity} is too small to retain the recordings including {path}")
        for step in steps:
            _validate_transition(step, buffer, duty_limits)
            buffer.add(step["obs"], step["next_obs"], step["action"], step["reward"], step["terminated"])
        loaded += len(steps)
    return loaded
