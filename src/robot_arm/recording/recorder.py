import os
import numpy as np
from typing import Dict, List, Any
from PIL import Image
from robot_arm.robot_schema import CAMERA_NAMES, CARTESIAN_ACTION_NAMES, MOTOR_ORDER


class EpisodeRecorder:
    """
    Records interactions for a single episode and saves them to disk.
    Images are saved as JPEGs or PNGs, and numeric data is saved in a JSON lines file
    or basic NPZ format for easy loading into dataset wrappers later.
    """

    def __init__(
        self,
        output_dir: str,
        cfg: Any,
        episode_name: str,
    ):
        self.output_dir = output_dir
        self.episode_dir = os.path.join(output_dir, episode_name)
        self.external_camera_images_dir = os.path.join(self.episode_dir, "images", "external_camera")
        self.wrist_camera_images_dir = os.path.join(self.episode_dir, "images", "wrist_camera")
        self.jpeg_quality = cfg.camera.jpeg_quality
        self.joint_steps_per_cartesian_action = cfg.control.frequencies.joint // cfg.control.frequencies.cartesian
        self.record_sim_state = cfg.runtime.record_sim_state
        self.record_policy_debug = cfg.runtime.record_policy_debug
        self.capture_camera = cfg.runtime.capture_camera
        self.motor_order = MOTOR_ORDER
        self.states: List[Dict[str, Any]] = []
        self.transitions: List[Dict[str, Any]] = []
        self.dense_trajectory_buffer: List[Dict[str, Any]] = []
        self.waypoints = None

        os.makedirs(self.episode_dir, exist_ok=True)
        if self.capture_camera:
            os.makedirs(self.external_camera_images_dir, exist_ok=True)
            os.makedirs(self.wrist_camera_images_dir, exist_ok=True)

    def save_waypoints(self, waypoints: list[np.ndarray]):
        """
        Record the raw waypoints provided by the high level policy layout.
        """
        self.waypoints = waypoints.copy()

    def record_transition(
        self,
        state_idx: int,
        state,
        reward: float,
        cartesian_action,
        teacher_action,
        images: dict[str, np.ndarray] | None,
        vla_input_state: np.ndarray,
        primitive_prompt: str,
        primitive_index: int,
    ):
        assert teacher_action.cartesian_action.shape == (len(CARTESIAN_ACTION_NAMES),)
        self.states.append(
            self._make_state(
                state_idx=state_idx,
                primitive_index=primitive_index,
                obs=state.observation,
                sensor_state=state.sensor_state,
                pose=state.end_effector_pose,
                sim_state=state.sim_state if self.record_sim_state else None,
                images=images,
                grasp_confirmed=state.grasp_confirmed,
            )
        )
        self.transitions.append(
            {
                "step": state_idx,
                "primitive_prompt": primitive_prompt,
                "vla_input_state": vla_input_state.copy(),
                "reward": float(reward),
                "cartesian_action": cartesian_action.cartesian_action.copy(),
                "teacher_cartesian_action": teacher_action.cartesian_action.copy(),
                "teacher_completes_active_primitive": bool(teacher_action.completes_active_primitive),
                "teacher_completion_score": float(teacher_action.diagnostics["teacher_completion_score"]),
                "desired_gripper_duty": float(cartesian_action.desired_gripper_duty),
                "desired_gripper_duty_active": bool(cartesian_action.desired_gripper_duty_active),
                "teacher_desired_gripper_duty": float(teacher_action.desired_gripper_duty),
                "teacher_desired_gripper_duty_active": bool(teacher_action.desired_gripper_duty_active),
                "diagnostics": {
                    **teacher_action.diagnostics,
                    **cartesian_action.diagnostics,
                    "teacher_completes_active_primitive": teacher_action.completes_active_primitive,
                },
                "completes_active_primitive": bool(cartesian_action.completes_active_primitive),
                "dense_trajectory": self.dense_trajectory_buffer.copy(),
            }
        )
        self.dense_trajectory_buffer.clear()

    def record_final_state(
        self,
        state,
        state_idx: int,
        primitive_index: int,
        images: dict[str, np.ndarray] | None,
    ):
        self.states.append(
            self._make_state(
                state_idx=state_idx,
                primitive_index=primitive_index,
                obs=state.observation,
                sensor_state=state.sensor_state,
                pose=state.end_effector_pose,
                sim_state=state.sim_state if self.record_sim_state else None,
                images=images,
                grasp_confirmed=state.grasp_confirmed,
            )
        )

    def _make_state(self, state_idx, primitive_index, obs, sensor_state, pose, sim_state, images, grasp_confirmed):
        return {
            "grasp_confirmed": bool(grasp_confirmed),
            "step": state_idx,
            "primitive_index": primitive_index,
            "image_paths": self._save_images(state_idx, images) if self.capture_camera else None,
            "joint_positions": obs["joint_positions"].copy(),
            "joint_velocities": obs["joint_velocities"].copy(),
            "sensor_state": sensor_state,
            "end_effector_pose": pose.as_10d(),
            "sim_state": sim_state,
        }

    def append_joint_transition(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        reward_breakdown: Dict[str, float],
        terminated: bool,
        duty_action: np.ndarray,
        state,
        next_state,
    ):
        """
        Record a single joint RL transition step in the environment.
        Currently, this method just caches dense steps on the most recently added frame
        so that when we save, we can view the whole micro-trajectory that occurred during the high-level step.
        """
        if len(self.dense_trajectory_buffer) >= self.joint_steps_per_cartesian_action:
            raise RuntimeError(f"Dense trajectory buffer overflow! Max is {self.joint_steps_per_cartesian_action} joint steps per cartesian action.")

        self.dense_trajectory_buffer.append(
            {
                "obs": {k: v.copy() for k, v in obs.items() if isinstance(v, np.ndarray)},
                "next_obs": {k: v.copy() for k, v in next_obs.items() if isinstance(v, np.ndarray)},
                "action": action.copy(),
                "requested_duty": duty_action.copy(),
                "reward": float(reward),
                "reward_breakdown": {key: float(value) for key, value in reward_breakdown.items()},
                "terminated": terminated,
                "end_effector_pose": state.end_effector_pose.as_10d(),
                "next_end_effector_pose": next_state.end_effector_pose.as_10d(),
            }
        )

    def _save_images(self, state_idx: int, images: dict[str, np.ndarray]) -> dict[str, str]:
        assert tuple(images) == CAMERA_NAMES, f"Expected cameras {CAMERA_NAMES}, got {tuple(images)}."
        external_camera_image_path = f"images/external_camera/frame_{state_idx:04d}.jpg"
        wrist_camera_image_path = f"images/wrist_camera/frame_{state_idx:04d}.jpg"
        Image.fromarray(images["external_camera"]).save(
            os.path.join(self.episode_dir, external_camera_image_path),
            format="JPEG",
            quality=self.jpeg_quality,
        )
        Image.fromarray(images["wrist_camera"]).save(
            os.path.join(self.episode_dir, wrist_camera_image_path),
            format="JPEG",
            quality=self.jpeg_quality,
        )
        return {
            "external_camera": external_camera_image_path,
            "wrist_camera": wrist_camera_image_path,
        }

    def save(self):
        """
        Write the buffered data to disk as a compressed .npz archive.
        """
        episode_path = os.path.join(self.episode_dir, "episode.npz")

        # TODO(lerobot): Review this state/transition layout against LeRobot's dataset schema.

        data_dict = {
            "box_gripped": np.array([s["grasp_confirmed"] for s in self.states], dtype=bool),
            "step": np.array([s["step"] for s in self.states], dtype=np.int32),
            "primitive_prompt": np.array([t["primitive_prompt"] for t in self.transitions], dtype=str),
            "primitive_index": np.array([s["primitive_index"] for s in self.states], dtype=np.int32),
            "vla_input_state": np.array(
                [t["vla_input_state"] for t in self.transitions],
                dtype=np.float32,
            ),
            "end_effector_pose": np.array(
                [s["end_effector_pose"] for s in self.states],
                dtype=np.float32,
            ),
            "joint_positions": np.array(
                [s["joint_positions"] for s in self.states],
                dtype=np.float32,
            ),
            "joint_velocities": np.array(
                [s["joint_velocities"] for s in self.states],
                dtype=np.float32,
            ),
            "sensor_read_started_ns": np.array([s["sensor_state"]["read_started_ns"] for s in self.states], dtype=np.int64),
            "sensor_read_completed_ns": np.array([s["sensor_state"]["read_completed_ns"] for s in self.states], dtype=np.int64),
            "sensor_sample_time_ns": np.array([s["sensor_state"]["sample_time_ns"] for s in self.states], dtype=np.int64),
            "cartesian_action": np.array([t["cartesian_action"] for t in self.transitions], dtype=object),
            "teacher_cartesian_action": np.asarray([t["teacher_cartesian_action"] for t in self.transitions], dtype=np.float32).reshape(-1, len(CARTESIAN_ACTION_NAMES)),
            "teacher_completes_active_primitive": np.asarray([t["teacher_completes_active_primitive"] for t in self.transitions], dtype=bool),
            "teacher_completion_score": np.asarray([t["teacher_completion_score"] for t in self.transitions], dtype=np.float32),
            "desired_gripper_duty": np.asarray([t["desired_gripper_duty"] for t in self.transitions], dtype=np.float32),
            "desired_gripper_duty_active": np.asarray([t["desired_gripper_duty_active"] for t in self.transitions], dtype=bool),
            "teacher_desired_gripper_duty": np.asarray([t["teacher_desired_gripper_duty"] for t in self.transitions], dtype=np.float32),
            "teacher_desired_gripper_duty_active": np.asarray([t["teacher_desired_gripper_duty_active"] for t in self.transitions], dtype=bool),
            "cartesian_action_diagnostics": np.array([t["diagnostics"] for t in self.transitions], dtype=object),
            "completes_active_primitive": np.array([t["completes_active_primitive"] for t in self.transitions], dtype=bool),
            "reward": np.array([t["reward"] for t in self.transitions], dtype=np.float32),
            "dense_trajectory": np.array([t["dense_trajectory"] for t in self.transitions], dtype=object),
        }

        if self.capture_camera:
            data_dict["external_camera_image_path"] = np.array(
                [state["image_paths"]["external_camera"] for state in self.states],
                dtype=str,
            )
            data_dict["wrist_camera_image_path"] = np.array(
                [state["image_paths"]["wrist_camera"] for state in self.states],
                dtype=str,
            )

        sensor_names = ("Present_Temperature", "Present_Load", "Present_Voltage")
        for sensor_name in sensor_names:
            data_dict[f"sensor_{sensor_name.removeprefix('Present_').lower()}"] = np.array(
                [[s["sensor_state"][sensor_name][motor] for motor in self.motor_order] for s in self.states],
                dtype=np.float32,
            )

        if self.record_sim_state:
            data_dict["body_pos"] = self.states[0]["sim_state"]["body_pos"]
            data_dict["geom_matid"] = self.states[0]["sim_state"]["geom_matid"]
            data_dict["qpos"] = np.array([s["sim_state"]["qpos"] for s in self.states], dtype=np.float32)
            data_dict["qvel"] = np.array([s["sim_state"]["qvel"] for s in self.states], dtype=np.float32)

        if self.waypoints is not None:
            data_dict["waypoints"] = self.waypoints

        np.savez_compressed(episode_path, **data_dict)
        print(f"Saved to: {episode_path}")
