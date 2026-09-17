from pathlib import Path

from tqdm import tqdm

from robot_arm.episode_runner import EpisodeRunner
from robot_arm.policies.primitive_generator import ScriptedPrimitiveGeneratorPolicy
from robot_arm.recording.recorder import EpisodeRecorder


def collect_episodes(cfg, env, joint_policy, cartesian_policy, run_dir: Path, num_episodes: int) -> Path:
    primitive_generator_policy = ScriptedPrimitiveGeneratorPolicy(cfg)
    recordings_dir = run_dir / "recordings"
    for episode_index in range(num_episodes):
        episode_name = f"episode_{episode_index:04d}"
        recorder = EpisodeRecorder(str(recordings_dir), cfg, episode_name)
        with tqdm(
            total=int(cfg.control.max_seconds * cfg.control.frequencies.cartesian),
            desc=f"Episode {episode_index + 1}/{num_episodes}",
            unit="action",
        ) as progress:
            runner = EpisodeRunner(
                cfg=cfg,
                env=env,
                joint_policy=joint_policy,
                primitive_policy=primitive_generator_policy,
                cartesian_policy=cartesian_policy,
                training=False,
                recorder=recorder,
                replay_buffer=None,
                metrics_queue=None,
                weights_queue=None,
                progress=progress,
            )
            runner.run_episode(generate_primitives=True)
    return recordings_dir
