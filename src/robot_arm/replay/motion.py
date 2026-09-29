import numpy as np

from robot_arm.robot_schema import POLICY_OBSERVATION_NAMES, STATE_TCP_VELOCITY_SLICE


def build_motion_comparison(policy, trajectories, cfg):
    samples = [sample for trajectory in trajectories for sample in trajectory]
    observations = {name: np.stack([sample["obs"][name] for sample in samples]) for name in POLICY_OBSERVATION_NAMES}
    actions = np.stack([sample["action"] for sample in samples])
    predicted = policy.predict_forward_motion(observations, actions)[:, 6:9]
    actual = np.stack([sample["next_obs"]["state"][STATE_TCP_VELOCITY_SLICE][:3] for sample in samples])
    scale = float(cfg.waypoint.position_speed_meters_per_second)
    return {
        "times": (np.arange(len(samples)) / cfg.control.frequencies.joint).tolist(),
        "predicted": (predicted * scale).tolist(),
        "actual": (actual * scale).tolist(),
        "error": ((predicted - actual) * scale).tolist(),
    }
