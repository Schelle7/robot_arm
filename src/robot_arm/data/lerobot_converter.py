import os
import glob
import logging
import sys
import tempfile
from contextlib import contextmanager

import av
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from robot_arm.robot_schema import CARTESIAN_ACTION_NAMES, INITIAL_POSE_NAMES, TARGET_OFFSET_NAMES


@contextmanager
def _quiet_native_stderr():
    original_stderr = os.dup(2)
    with tempfile.TemporaryFile() as captured_stderr:
        os.dup2(captured_stderr.fileno(), 2)
        try:
            yield
        except BaseException:
            os.dup2(original_stderr, 2)
            captured_stderr.seek(0)
            sys.stderr.buffer.write(captured_stderr.read())
            sys.stderr.flush()
            raise
        finally:
            os.dup2(original_stderr, 2)
            os.close(original_stderr)


def _validate_and_load_configs(episodes: list[str]):
    # The input epsiodes path looks like: outputs/collect_data/.../recordings/waypoint_dataset_01/episode.npz
    # We need to traverse up 3 levels to reach the root outputs/... directory where .hydra lives.
    run_dirs = {os.path.dirname(os.path.dirname(os.path.dirname(ep))) for ep in episodes}
    ref_cfg = None
    ref_dir = None

    for run_dir in run_dirs:
        cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing config.yaml at {cfg_path}")

        cfg = OmegaConf.load(cfg_path)
        if ref_cfg is None:
            ref_cfg = cfg
            ref_dir = run_dir
        else:
            # Check agreement on critical dimensions
            if (
                cfg.camera.height != ref_cfg.camera.height
                or cfg.camera.width != ref_cfg.camera.width
                or cfg.waypoint.trajectory_length != ref_cfg.waypoint.trajectory_length
                or cfg.waypoint.trajectory_dim != ref_cfg.waypoint.trajectory_dim
            ):
                raise ValueError(f"Configuration mismatch detected between {ref_dir} and {run_dir}. " f"Datasets must have identical dimensions.")
    return ref_cfg


def convert_to_lerobot(source_dir: str, target_dir: str, fps: int):
    """
    Parses flat .npz tracking outputs and corresponding jpegs,
    and converts them into LeRobot/Hugging Face format using LeRobotDataset.create().
    """
    av.logging.set_level(av.logging.ERROR)
    logging.getLogger("lerobot.datasets.video_utils").setLevel(logging.WARNING)

    # Find all episodes in the source directory
    episodes = sorted(glob.glob(os.path.join(source_dir, "**", "episode.npz"), recursive=True))
    if not episodes:
        raise FileNotFoundError(f"No episode.npz files found in {source_dir}")

    # Discover and validate run configurations
    ref_cfg = _validate_and_load_configs(episodes)
    path_length = int(ref_cfg.waypoint.trajectory_length)
    cartesian_action_dim = int(ref_cfg.waypoint.trajectory_dim)
    assert cartesian_action_dim == len(CARTESIAN_ACTION_NAMES)
    action_names = [
        f"step_{path_index:02d}_{action_name}"
        for path_index in range(path_length)
        for action_name in CARTESIAN_ACTION_NAMES
    ]

    # 1. Define the dataset features dynamically based on verified config
    features = {
        "observation.images.camera1": {
            "dtype": "video",
            "shape": (ref_cfg.camera.height, ref_cfg.camera.width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (15,),
            "names": INITIAL_POSE_NAMES + TARGET_OFFSET_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (path_length * cartesian_action_dim,),
            "names": action_names,
        },
    }

    # Use the target_dir name as the repo_id (e.g. "robot_arm_vla_dataset")
    repo_id = os.path.basename(os.path.normpath(target_dir))

    # Create the dataset root
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=target_dir,
        features=features,
        use_videos=True,
        vcodec="h264",
    )

    print(f"Found {len(episodes)} episodes. Converting to LeRobot dataset at {target_dir}...")

    # 3. Process each episode
    for ep_idx, ep_path in enumerate(tqdm(episodes)):
        ep_dir = os.path.dirname(ep_path)

        # Load the numeric data
        data = np.load(ep_path, allow_pickle=True)
        num_frames = len(data["step"])

        for frame_idx in range(num_frames - 1):
            # Load the corresponding image. The NPZ contains the relative path.
            # We align observation t with action t (which is technically recorded at step+1 in our loop).
            image_rel_path = data["image_path"][frame_idx]
            image_abs_path = os.path.join(ep_dir, image_rel_path)

            img = Image.open(image_abs_path).convert("RGB")

            # State t (15D VLA input state)
            state = torch.tensor(data["vla_input_state"][frame_idx], dtype=torch.float32)

            # Action t is selected from state t.
            action_raw = np.asarray(
                data["cartesian_action_path"][frame_idx],
                dtype=np.float32,
            ).reshape(-1)
            action = torch.from_numpy(action_raw)

            task = str(data["primitive_prompt"][frame_idx])

            # Add frame to the dataset
            dataset.add_frame(
                {
                    "observation.images.camera1": img,
                    "observation.state": state,
                    "action": action,
                    "task": task,
                }
            )

        # Complete the episode
        with _quiet_native_stderr():
            dataset.save_episode()

    print("Dataset conversion complete!")
