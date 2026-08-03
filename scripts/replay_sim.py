import os
import glob
import numpy as np
import mujoco
import mujoco.viewer
import hydra
from omegaconf import DictConfig
from robot_arm.pose import Pose
from robot_arm.backends.sim_arm import (
    build_desired_poses,
    update_tcp_debug_user_scene,
    update_waypoint_debug_user_scene,
    update_desired_pose_debug_user_scene,
)


def find_latest_episode():
    outputs_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "outputs")
    )
    search_pattern = os.path.join(
        outputs_dir, "rollout_waypoint", "*", "*", "**", "episode.npz"
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


@hydra.main(version_base=None, config_path="../conf", config_name="replay")
def main(cfg: DictConfig):
    episode_path = cfg.episode_path
    if episode_path is None:
        episode_path = find_latest_episode()
        if episode_path is None:
            print("Could not find any episode.npz files in outputs/rollout_waypoint.")
            return

    print(f"Loading episode from: {episode_path}")

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
    num_frames = len(qpos_recording)

    print(f"Loaded {num_frames} frames of simulation state.")

    # Load the MuJoCo model
    try:
        model = mujoco.MjModel.from_xml_path(cfg.model_path)
        mdata = mujoco.MjData(model)
    except Exception as e:
        print(f"Failed to load MuJoCo model from {cfg.model_path}: {e}")
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
            current_frame[0] = min(current_frame[0] + 1, num_frames - 1)
            print(f"Frame: {current_frame[0]}/{num_frames-1}")
        elif keycode == 263:  # Left arrow
            current_frame[0] = max(current_frame[0] - 1, 0)
            print(f"Frame: {current_frame[0]}/{num_frames-1}")
        elif keycode == 32:  # Space
            auto_play[0] = not auto_play[0]
            print(f"Auto-play: {'ON' if auto_play[0] else 'OFF'}")

    # Pre-set the initial frame
    mdata.qpos[:] = qpos_recording[0]
    mdata.qvel[:] = qvel_recording[0]

    wps = data["waypoints"]
    desired_actions = data["high_level_delta_action"]
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
                current_frame[0] = (current_frame[0] + 1) % num_frames

            mdata.qpos[:] = qpos_recording[current_frame[0]]
            mdata.qvel[:] = qvel_recording[current_frame[0]]
            mujoco.mj_forward(model, mdata)
            update_tcp_debug_user_scene(viewer_inst.user_scn, model, mdata)
            update_waypoint_debug_user_scene(viewer_inst.user_scn, wps, 0)
            desired_start_pose = Pose.from_10d(recorded_poses[current_frame[0]])
            desired_poses = build_desired_poses(
                desired_start_pose,
                desired_actions[current_frame[0]],
                cfg.runtime.desired_path_visual_exaggeration_factor,
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
