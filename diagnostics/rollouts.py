import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from robot_arm.run_paths import find_runs

ROLLOUT_ROOT = Path("outputs/rollout")


@dataclass
class Rollout:
    run_dir: Path
    config: DictConfig
    episodes: dict[Path, dict[str, np.ndarray]]
    servo_configuration: dict[str, Any] | None
    git_status: str | None
    log: str | None

    @property
    def job(self) -> str:
        return self.run_dir.parent.parent.name

    @property
    def started(self) -> str:
        return f"{self.run_dir.parent.name} {self.run_dir.name}"

    @property
    def model_path(self) -> Path:
        return self.run_dir / "model" / Path(self.config.model_path).name


def find_rollouts(limit: int, root: Path) -> list[Path]:
    """
    Run directories, newest first. Keyed on the hydra config rather than the recording, so a run
    that tripped the safety wrapper before it ever reached save() is still listed. Those are the
    interesting ones.
    """
    runs = find_runs([path for path in root.glob("*") if path.is_dir()])
    return [run for run in runs if (run / ".hydra/config.yaml").is_file()][:limit]


def load_rollout(run_dir: Path | str) -> Rollout:
    run_dir = Path(run_dir)
    servo_path = run_dir / "servo_configuration.json"
    status_path = run_dir / "git" / "status.txt"
    log_path = next(run_dir.glob("*.log"), None)

    return Rollout(
        run_dir=run_dir,
        config=OmegaConf.load(run_dir / ".hydra" / "config.yaml"),
        episodes={path: _load_episode(path) for path in sorted(run_dir.glob("**/episode.npz"))},
        servo_configuration=json.loads(servo_path.read_text()) if servo_path.exists() else None,
        git_status=status_path.read_text() if status_path.exists() else None,
        log=log_path.read_text() if log_path else None,
    )


def dense_steps(data: dict[str, np.ndarray]) -> list[dict]:
    """
    Every joint step in the episode, flattened out of the per-cartesian-action chunks the
    recorder nests them in.
    """
    return [step for chunk in data["dense_trajectory"] for step in chunk]


def stack_dense(steps: list[dict], key: str) -> np.ndarray:
    return np.array([step[key] for step in steps], dtype=np.float32)


def _load_episode(episode_path: Path) -> dict[str, np.ndarray]:
    with np.load(episode_path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}
