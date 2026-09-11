import os
import glob
import json
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
from robot_arm.geometry.pose import Pose
from robot_arm.robot_schema import CAMERA_NAMES, CARTESIAN_ACTION_NAMES, CURRENT_POSE_NAMES, DUTY_NAMES, PRIMITIVE_COMPLETION, TARGET_OFFSET_NAMES


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


def _validate_and_load_configs(episodes: list[str], fps: int):
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
        recorded_fps = cfg.control.frequencies.cartesian
        if recorded_fps != fps:
            raise ValueError(f"Recording frequency mismatch in {run_dir}: recorded {recorded_fps} FPS, conversion requested {fps} FPS.")
        if ref_cfg is None:
            ref_cfg = cfg
            ref_dir = run_dir
        else:
            # Check agreement on critical dimensions
            if (
                cfg.camera.cameras.external_camera.height != ref_cfg.camera.cameras.external_camera.height
                or cfg.camera.cameras.external_camera.width != ref_cfg.camera.cameras.external_camera.width
                or cfg.camera.cameras.wrist_camera.height != ref_cfg.camera.cameras.wrist_camera.height
                or cfg.camera.cameras.wrist_camera.width != ref_cfg.camera.cameras.wrist_camera.width
                or cfg.waypoint.cartesian_action_dim != ref_cfg.waypoint.cartesian_action_dim
            ):
                raise ValueError(f"Configuration mismatch detected between {ref_dir} and {run_dir}. " f"Datasets must have identical dimensions.")
    return ref_cfg


def _build_dataset_features(ref_cfg):
    cartesian_action_dim = int(ref_cfg.waypoint.cartesian_action_dim)
    assert cartesian_action_dim == len(CARTESIAN_ACTION_NAMES)
    assert tuple(ref_cfg.camera.cameras) == CAMERA_NAMES
    features = {
        "observation.images.external_camera": {
            "dtype": "video",
            "shape": (
                ref_cfg.camera.cameras.external_camera.height,
                ref_cfg.camera.cameras.external_camera.width,
                3,
            ),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist_camera": {
            "dtype": "video",
            "shape": (
                ref_cfg.camera.cameras.wrist_camera.height,
                ref_cfg.camera.cameras.wrist_camera.width,
                3,
            ),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(CURRENT_POSE_NAMES) + len(TARGET_OFFSET_NAMES) + len(DUTY_NAMES),),
            "names": CURRENT_POSE_NAMES + TARGET_OFFSET_NAMES + DUTY_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (cartesian_action_dim,),
            "names": list(CARTESIAN_ACTION_NAMES),
        },
        PRIMITIVE_COMPLETION: {
            "dtype": "float32",
            "shape": (1,),
            "names": [PRIMITIVE_COMPLETION],
        },
    }
    return features


def _reconstruct_vla_input_state(data, frame_idx: int) -> torch.Tensor:
    current_pose = Pose.from_10d(data["end_effector_pose"][frame_idx])
    recorded_state = data["vla_input_state"][frame_idx]
    target_offset_flag = float(recorded_state[len(CURRENT_POSE_NAMES) + len(TARGET_OFFSET_NAMES) - 1])
    gripper_duty = float(recorded_state[-1])
    if target_offset_flag == 1.0:
        primitive_index = int(data["primitive_index"][frame_idx])
        target_pose = Pose.from_10d(data["waypoints"][primitive_index])
        target_offset = current_pose.delta_to(target_pose)
    else:
        target_offset = np.zeros(7, dtype=np.float32)
    state = np.concatenate([current_pose.as_7d(), target_offset, [target_offset_flag], [gripper_duty]]).astype(np.float32)
    return torch.from_numpy(state)


def grip_filter_result(data, fps: int, min_gripped_seconds: float) -> dict:
    prompts = data["primitive_prompt"]
    closing = prompts == "close gripper"
    lifting = prompts == "lift object"
    transporting = np.char.startswith(prompts, "move ") & np.char.endswith(prompts, " tile")
    lowering = np.char.startswith(prompts, "lower ") & np.char.endswith(prompts, " tile")
    gripped = data["box_gripped"][:-1]
    assert gripped.shape == prompts.shape, "Expected one gripped state per transition plus one final state."
    holding = closing | lifting | transporting | lowering
    gripped_seconds = float(np.count_nonzero(gripped & holding) / fps)
    if not np.any(closing):
        accepted, reason = True, "No close-gripper primitive; grip filter does not apply."
    elif not np.any(lifting):
        accepted, reason = False, "Closing occurred but no lift-object primitive was recorded."
    elif gripped_seconds < min_gripped_seconds:
        accepted, reason = False, "Confirmed grasp time during closing/lifting/transport/lowering is below the minimum."
    else:
        accepted, reason = True, "Confirmed grasp time meets the minimum."
    return {
        "accepted": accepted,
        "reason": reason,
        "confirmed_grasp_seconds": gripped_seconds,
        "required_grasp_seconds": min_gripped_seconds,
        "recorded_seconds": len(prompts) / fps,
        "has_closing": bool(np.any(closing)),
        "has_lifting": bool(np.any(lifting)),
    }


def filter_failed_grips(data, fps: int, min_gripped_seconds: float) -> bool:
    return grip_filter_result(data, fps, min_gripped_seconds)["accepted"]


def write_conversion_report(path: str, report: dict) -> None:
    with open(path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
        report_file.write("\n")


def convert_to_lerobot(
    source_dir: str,
    target_dir: str,
    fps: int,
    min_gripped_seconds: float,
    video_files_size_in_mb: float,
):
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
    ref_cfg = _validate_and_load_configs(episodes, fps)
    features = _build_dataset_features(ref_cfg)

    accepted_episodes = []
    episode_results = {}
    for ep_path in episodes:
        with np.load(ep_path, allow_pickle=True) as data:
            result = grip_filter_result(data, fps, min_gripped_seconds)
            result["source_episode"] = os.path.abspath(ep_path)
            result["conversion_status"] = "pending" if result["accepted"] else "not_selected"
            episode_results[ep_path] = result
            if result["accepted"]:
                accepted_episodes.append(ep_path)
            else:
                print(f"Rejected {ep_path}: {result['reason']} ({result['confirmed_grasp_seconds']:.2f}s confirmed)")
    report_path = os.path.abspath(target_dir) + ".conversion_report.json"
    report = {
        "source_dir": os.path.abspath(source_dir),
        "target_dir": os.path.abspath(target_dir),
        "fps": fps,
        "video_files_size_in_mb": video_files_size_in_mb,
        "accepted_count": len(accepted_episodes),
        "rejected_count": len(episodes) - len(accepted_episodes),
        "conversion_complete": False,
        "episodes": list(episode_results.values()),
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    write_conversion_report(report_path, report)
    print(f"Conversion report: {report_path}")
    print(f"Grip filter: accepted {len(accepted_episodes)}, rejected {len(episodes) - len(accepted_episodes)} recordings.")
    if not accepted_episodes:
        raise ValueError("All recordings were rejected by the failed-grip filter.")
    episodes = accepted_episodes

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
    dataset.meta.update_chunk_settings(video_files_size_in_mb=video_files_size_in_mb)

    print(f"Found {len(episodes)} episodes. Converting to LeRobot dataset at {target_dir}...")

    # 3. Process each episode
    for ep_idx, ep_path in enumerate(tqdm(episodes)):
        episode_results[ep_path]["conversion_status"] = "converting"
        write_conversion_report(report_path, report)
        ep_dir = os.path.dirname(ep_path)

        # Load the numeric data
        data = np.load(ep_path, allow_pickle=True)
        num_frames = len(data["step"])

        for frame_idx in range(num_frames - 1):
            external_camera_image = Image.open(
                os.path.join(ep_dir, data["external_camera_image_path"][frame_idx])
            ).convert("RGB")
            wrist_camera_image = Image.open(
                os.path.join(ep_dir, data["wrist_camera_image_path"][frame_idx])
            ).convert("RGB")

            state = _reconstruct_vla_input_state(data, frame_idx)

            # Action t is selected from state t.
            action_raw = np.asarray(
                data["cartesian_action"][frame_idx],
                dtype=np.float32,
            ).reshape(-1)
            action = torch.from_numpy(action_raw)

            task = str(data["primitive_prompt"][frame_idx])
            primitive_completion = np.asarray(
                [data["completes_active_primitive"][frame_idx]],
                dtype=np.float32,
            )

            # Add frame to the dataset
            frame = {
                "observation.images.external_camera": external_camera_image,
                "observation.images.wrist_camera": wrist_camera_image,
                "observation.state": state,
                "action": action,
                PRIMITIVE_COMPLETION: primitive_completion,
                "task": task,
            }
            dataset.add_frame(frame)

            if frame_idx == num_frames - 2 or data["primitive_index"][frame_idx + 1] != data["primitive_index"][frame_idx]:
                with _quiet_native_stderr():
                    dataset.save_episode()

        episode_results[ep_path]["conversion_status"] = "converted"
        write_conversion_report(report_path, report)

    dataset.finalize()
    report["conversion_complete"] = True
    write_conversion_report(report_path, report)
    print("Dataset conversion complete!")
