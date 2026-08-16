from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

from robot_arm.replay import load_replay_recording, recorded_model_path
from robot_arm.replay_viewer import ReplayViewer


def display_replay(display_lines, warnings):
    print("\033[H\033[J", end="")
    print("\n".join(display_lines), flush=True)
    if warnings:
        print("\n".join(warnings), flush=True)


def format_rollout_age(episode_path: str) -> str:
    rollout_directory = Path(episode_path).resolve().parents[2]
    rollout_time = datetime.strptime(
        f"{rollout_directory.parent.name} {rollout_directory.name}",
        "%Y-%m-%d %H-%M-%S",
    )
    age_seconds = max(0, int((datetime.now() - rollout_time).total_seconds()))
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
    recorded_mid_level_hz = recorded_cfg.control.frequencies.mid_level
    recorded_low_level_hz = recorded_cfg.control.frequencies.low_level
    print("Recorded frequencies: " f"mid-level={recorded_mid_level_hz} Hz, " f"low-level={recorded_low_level_hz} Hz")
    viewer = ReplayViewer(
        recorded_cfg,
        data,
        recorded_model_path(episode_path, recorded_cfg),
        window_title,
    )
    viewer.run(display_replay)


if __name__ == "__main__":
    main()
