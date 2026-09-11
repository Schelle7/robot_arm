from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

from robot_arm.replay.recording import load_replay_recording, recorded_model_path, rollout_timestamp
from robot_arm.replay.server import ReplayServer
from robot_arm.replay.viewer import ReplayViewer


def format_rollout_age(episode_path: str) -> str:
    age_seconds = max(0, int((datetime.now() - rollout_timestamp(episode_path)).total_seconds()))
    hours, remainder = divmod(age_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s old"
    if minutes:
        return f"{minutes}m {seconds}s old"
    return f"{seconds}s old"


@hydra.main(version_base=None, config_path="../conf", config_name="replay")
def main(cfg: DictConfig):
    episode_path, recorded_cfg, data = load_replay_recording(cfg)
    window_title = f"Replaying {format_rollout_age(episode_path)} rollout: {Path(episode_path).resolve()}"
    recorded_cartesian_hz = recorded_cfg.control.frequencies.cartesian
    recorded_joint_hz = recorded_cfg.control.frequencies.joint
    duty_limits = {motor: float(recorded_cfg.servo.max_duty[motor] / recorded_cfg.servo.full_scale_duty) for motor in recorded_cfg.servo.max_duty}
    print(f"Recorded frequencies: cartesian={recorded_cartesian_hz} Hz, joint={recorded_joint_hz} Hz")
    viewer = ReplayViewer(
        recorded_cfg,
        data,
        recorded_model_path(episode_path, recorded_cfg),
        window_title,
    )
    server = ReplayServer(
        cfg.display_port,
        viewer.num_states,
        recorded_cartesian_hz,
        episode_path,
        cfg.branch_request_path,
        duty_limits,
        recorded_joint_hz,
        viewer.action_history,
        viewer.has_sim_state,
    )
    server.start()
    viewer.run(server.display, server.take_commands)


if __name__ == "__main__":
    main()
