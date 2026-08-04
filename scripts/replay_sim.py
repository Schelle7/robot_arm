import os
import glob
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import hydra
from omegaconf import DictConfig, OmegaConf
from robot_arm.pose import Pose
from robot_arm.backends.sim_arm import (
    build_desired_poses,
    update_tcp_debug_user_scene,
    update_waypoint_debug_user_scene,
    update_desired_pose_debug_user_scene,
)


def find_latest_episode():
    outputs_dir = Path(__file__).resolve().parents[1] / "outputs"
    search_pattern = os.path.join(
        str(outputs_dir), "rollout_waypoint", "*", "*", "**", "episode.npz"
    )
    files = glob.glob(search_pattern, recursive=True)

    if not files:
        return None

    def extract_datetime_key(filepath):
        parts = filepath.split(os.sep)
        idx = parts.index("rollout_waypoint")
        return (parts[idx + 1], parts[idx + 2])

    latest_episode = max(files, key=extract_datetime_key)
    return latest_episode


def load_recorded_config(episode_path: str) -> DictConfig:
    rollout_directory = Path(episode_path).resolve().parents[2]
    config_path = rollout_directory / ".hydra" / "config.yaml"
    return OmegaConf.load(config_path)


def recorded_model_path(episode_path: str, recorded_cfg: DictConfig) -> str:
    rollout_directory = Path(episode_path).resolve().parents[2]
    model_filename = Path(recorded_cfg.model_path).name
    return str(rollout_directory / "model" / model_filename)


def load_recorded_timing(episode_path: str):
    recorded_cfg = load_recorded_config(episode_path)
    recorded_low_level_hz = recorded_cfg.control.frequencies.low_level
    recorded_high_level_hz = recorded_cfg.control.frequencies.high_level
    if recorded_cfg.backend != "sim":
        raise ValueError(
            f"replay_sim.py requires a simulation recording, got {recorded_cfg.backend!r}."
        )
    return (
        recorded_cfg,
        recorded_low_level_hz,
        recorded_high_level_hz,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="replay")
def main(cfg: DictConfig):
    episode_path = cfg.episode_path
    if episode_path is None:
        episode_path = find_latest_episode()
        if episode_path is None:
            print("Could not find any episode.npz files in outputs/rollout_waypoint.")
            return

    print(f"Loading episode from: {episode_path}")
    (
        recorded_cfg,
        recorded_low_level_hz,
        recorded_high_level_hz,
    ) = load_recorded_timing(episode_path)
    print(
        "Recorded frequencies: "
        f"high-level={recorded_high_level_hz} Hz, "
        f"low-level={recorded_low_level_hz} Hz"
    )

    try:
        data = np.load(episode_path, allow_pickle=True)
    except Exception as e:
        print(f"Failed to load episode data: {e}")
        return

    if "qpos" not in data or "qvel" not in data:
        print("Error: The loaded episode does not contain 'qpos' or 'qvel' data.")
        print("Make sure it was recorded with 'record_sim_state: true'.")
        return

    qpos_recording = data["qpos"]
    qvel_recording = data["qvel"]
    desired_actions = data["high_level_delta_action"]
    num_states = len(qpos_recording)
    num_actions = len(desired_actions)
    if num_states != num_actions + 1:
        raise ValueError(
            "Replay recording must contain one more state than high-level actions."
        )

    print(f"Loaded {num_states} states and {num_actions} high-level actions.")

    # Load the MuJoCo model
    model_path = recorded_model_path(episode_path, recorded_cfg)
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        mdata = mujoco.MjData(model)
    except Exception as e:
        print(f"Failed to load MuJoCo model from {model_path}: {e}")
        return

    print("\nControls:")
    print("  Right Arrow : Next frame")
    print("  Left Arrow  : Previous frame")
    print("  Space       : Auto-play toggle")
    print("  Esc         : Quit\n")

    current_frame = [0]
    auto_play = [False]

    def key_callback(keycode):
        if keycode == 262:  # Right arrow
            current_frame[0] = min(current_frame[0] + 1, num_states - 1)
            print(f"State: {current_frame[0]}/{num_states - 1}")
        elif keycode == 263:  # Left arrow
            current_frame[0] = max(current_frame[0] - 1, 0)
            print(f"State: {current_frame[0]}/{num_states - 1}")
        elif keycode == 32:  # Space
            auto_play[0] = not auto_play[0]
            print(f"Auto-play: {'ON' if auto_play[0] else 'OFF'}")

    # Pre-set the initial frame
    mdata.qpos[:] = qpos_recording[0]
    mdata.qvel[:] = qvel_recording[0]

    wps = data["waypoints"]
    recorded_poses = data["privileged_end_effector_pose"]

    mujoco.mj_forward(model, mdata)
    with mujoco.viewer.launch_passive(
        model, mdata, key_callback=key_callback
    ) as viewer_inst:
        import time

        viewer_inst.opt.geomgroup[5] = 0

        while viewer_inst.is_running():
            step_start = time.time()

            if auto_play[0]:
                current_frame[0] = (current_frame[0] + 1) % num_states

            mdata.qpos[:] = qpos_recording[current_frame[0]]
            mdata.qvel[:] = qvel_recording[current_frame[0]]
            mujoco.mj_forward(model, mdata)
            update_tcp_debug_user_scene(viewer_inst.user_scn, model, mdata)
            update_waypoint_debug_user_scene(viewer_inst.user_scn, wps, 0)
            desired_poses = []
            if current_frame[0] < num_actions:
                desired_start_pose = Pose.from_10d(recorded_poses[current_frame[0]])
                desired_poses = build_desired_poses(
                    desired_start_pose,
                    desired_actions[current_frame[0]],
                )
            update_desired_pose_debug_user_scene(viewer_inst.user_scn, desired_poses)

            # Print frame index to console (to avoid spamming, only when it changes, but here we print if auto_playing or keyed)
            # Actually, to avoid too much spam, we just use a small sleep. The user can see it's moving.

            viewer_inst.sync()

            # Sleep to match a reasonable viewing rate (~10Hz for high level frames)
            time_until_next_step = 0.1 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
