import numpy as np
import queue
from omegaconf import DictConfig
from typing import Dict

from robot_arm.policies import CartesianAction, CartesianPolicy, waypoint_action_scale
from robot_arm.primitive_policy import ScriptedPrimitiveGeneratorPolicy
from robot_arm.primitives import ActionPrimitive
from robot_arm.envs.env import EnvironmentState, RobotEnv
from robot_arm.recorder import EpisodeRecorder


class EpisodeRunner:
    """
    Orchestrates the entire episode.
    Handles high/low level syncing, logging to recorder,
    and returns transitions for the training buffers.
    """

    def __init__(
        self,
        cfg: DictConfig,
        env: RobotEnv,
        low_level_policy,
        primitive_policy: ScriptedPrimitiveGeneratorPolicy,
        cartesian_policy: CartesianPolicy,
        training: bool,
        recorder: EpisodeRecorder,
        replay_buffer,
        metrics_queue,
        weights_queue,
    ):
        mid_level_hz = cfg.control.frequencies.mid_level
        low_level_hz = cfg.control.frequencies.low_level

        if low_level_hz % mid_level_hz != 0:
            raise ValueError(f"low_level_hz ({low_level_hz}) must be divisible by mid_level_hz ({mid_level_hz})")

        self.env = env
        self.chunk_size = low_level_hz // mid_level_hz
        self.low_level_policy = low_level_policy
        self.primitive_policy = primitive_policy
        self.cartesian_policy = cartesian_policy
        self.training = training
        self.recorder = recorder
        self.replay_buffer = replay_buffer
        self.metrics_queue = metrics_queue
        self.cfg = cfg
        self.weights_queue = weights_queue
        self.action_scale_radians_per_second = cfg.control.action_scale_radians_per_second

        self.cartesian_action_scale = waypoint_action_scale(
            cfg.waypoint.trajectory_length,
            low_level_hz,
            cfg.waypoint.position_speed_meters_per_second,
            cfg.waypoint.rotation_speed_radians_per_second,
            cfg.waypoint.gripper_speed_radians_per_second,
        )

        self.max_mid_level_steps = int(cfg.control.max_seconds * mid_level_hz)
        self.episode_low_level_step = 0
        self.training_chunk_count = 0

    def _prepare_primitives(self, generate_primitives: bool, start_pose):
        if generate_primitives:
            self.primitive_policy.generate(
                self.env.arm.model,
                self.env.arm.data,
                start_pose=start_pose,
            )

        if self.cfg.runtime.draw_waypoints:
            self.env.arm.draw_waypoints(self.primitive_policy.target_poses)

        if self.recorder:
            self.recorder.save_waypoints(self.primitive_policy.target_poses)

    def _initialize_debug_visualization(self):
        if self.cfg.runtime.draw_tcp:
            self.env.arm.draw_tcp()

    def _record_final_state(self, state: EnvironmentState, state_idx: int, primitive_index: int):
        if self.recorder:
            self.recorder.record_final_state(
                state_idx=state_idx,
                primitive_index=primitive_index,
                obs=state.observation,
                sensor_state=state.sensor_state,
                pose=state.privileged_state["end_effector_pose"],
                sim_state=state.privileged_state["sim_state"] if self.cfg.runtime.record_sim_state else None,
                image=self.env.read_camera(),
            )

    def _draw_desired_path(self, pose, action, active_waypoint_index):
        if self.cfg.runtime.draw_waypoints:
            self.env.arm.update_waypoint_index(active_waypoint_index)
            self.env.arm.draw_desired_path(pose, action)

    def _record_transition(
        self,
        state_idx: int,
        state: EnvironmentState,
        reward: float,
        image: np.ndarray,
        vla_input_state: np.ndarray,
        primitive_prompt: str,
        primitive_index: int,
        cartesian_action: CartesianAction,
    ):
        if self.recorder:
            self.recorder.record_transition(
                state_idx=state_idx,
                obs=state.observation,
                sensor_state=state.sensor_state,
                reward=reward,
                cartesian_action_path=cartesian_action.cartesian_action_path,
                pose=state.privileged_state["end_effector_pose"],
                sim_state=state.privileged_state["sim_state"] if self.cfg.runtime.record_sim_state else None,
                image=image,
                vla_input_state=vla_input_state,
                primitive_prompt=primitive_prompt,
                primitive_index=primitive_index,
                diagnostics=cartesian_action.diagnostics,
                completes_active_primitive=cartesian_action.completes_active_primitive,
            )

    def _publish_chunk_metrics(self, total_reward, chunk_reward_metrics):
        if self.training and self.metrics_queue:
            detailed_metrics = {"total_reward": total_reward}

            if self.cfg.training.detailed_metrics:
                detailed_metrics.update(chunk_reward_metrics)

            self.metrics_queue.add(detailed_metrics)

    def _build_policy_observation(
        self,
        observation: Dict[str, np.ndarray],
        cartesian_action_path: np.ndarray,
        start_positions: np.ndarray,
        chunk_step: int,
    ) -> Dict[str, np.ndarray]:
        policy_observation = dict(observation)
        policy_observation["joint_velocities"] = observation["joint_velocities"] / self.action_scale_radians_per_second
        policy_observation["cartesian_action_path"] = cartesian_action_path / self.cartesian_action_scale
        policy_observation["start_joint_positions"] = start_positions
        policy_observation["time_left"] = np.array(
            [(self.chunk_size - chunk_step) / self.chunk_size],
            dtype=np.float32,
        )
        return policy_observation

    def _record_low_level_transition(
        self,
        policy_observation: Dict[str, np.ndarray],
        next_policy_observation: Dict[str, np.ndarray],
        low_level_action: np.ndarray,
        reward: float,
        reward_breakdown: Dict[str, float],
        chunk_terminated: bool,
        state: EnvironmentState,
        next_state: EnvironmentState,
    ):
        if self.recorder and self.cfg.runtime.record_policy_debug:
            self.recorder.append_low_level_transition(
                policy_observation,
                next_policy_observation,
                low_level_action,
                reward,
                reward_breakdown,
                chunk_terminated,
                state,
                next_state,
            )

    def _add_to_replay_buffer(
        self,
        policy_observation: Dict[str, np.ndarray],
        next_policy_observation: Dict[str, np.ndarray],
        low_level_action: np.ndarray,
        reward: float,
        chunk_terminated: bool,
    ):
        if self.training and self.replay_buffer:
            self.replay_buffer.add(
                policy_observation,
                next_policy_observation,
                low_level_action,
                reward,
                chunk_terminated,
            )

    def _step_low_level(
        self,
        policy_observation: Dict[str, np.ndarray],
        cartesian_action_path: np.ndarray,
        chunk_start_pose: object,
        chunk_terminated: bool,
    ):
        low_level_action, _ = self.low_level_policy.predict(policy_observation, deterministic=not self.training)
        next_state, reward, reward_breakdown = self.env.step(
            low_level_action,
            cartesian_action_path,
            chunk_start_pose,
            chunk_terminated,
        )

        if self.cfg.runtime.draw_tcp:
            self.env.arm.draw_tcp()

        return low_level_action, next_state, reward, reward_breakdown

    def _update_chunk_reward_metrics(self, chunk_reward_metrics, reward_breakdown):
        if self.cfg.training.detailed_metrics:
            for key, value in reward_breakdown.items():
                if key not in chunk_reward_metrics:
                    chunk_reward_metrics[key] = []
                chunk_reward_metrics[key].append(value)
            if self.cfg.training.pose_delta_diagnostics_enabled:
                for key, value in self.env.pose_delta_diagnostics.items():
                    if key not in chunk_reward_metrics:
                        chunk_reward_metrics[key] = []
                    chunk_reward_metrics[key].append(value)

    def run_episode(self, generate_primitives: bool):
        try:
            self._run_episode(generate_primitives)
        finally:
            if self.recorder:
                self.recorder.save()

    def _run_episode(self, generate_primitives: bool):
        state = self.env.reset()

        self._prepare_primitives(generate_primitives, start_pose=state.privileged_state["end_effector_pose"])
        self._initialize_debug_visualization()

        completed_steps = 0
        while self.primitive_policy.has_next_primitive():
            primitive_index, primitive = self.primitive_policy.get_next_primitive(
                state.privileged_state["end_effector_pose"],
            )
            state, completed_steps, primitive_truncated = self.execute_primitive(
                state,
                primitive,
                primitive_index,
                completed_steps,
            )
            if primitive_truncated:
                break

        assert completed_steps > 0, "Episode completed without producing a cartesian transition."
        self._record_final_state(state, completed_steps, primitive_index)

    def execute_primitive(
        self,
        state: EnvironmentState,
        primitive: ActionPrimitive,
        primitive_index: int,
        completed_steps: int,
    ) -> tuple[EnvironmentState, int, bool]:
        while True:
            if completed_steps >= self.max_mid_level_steps:
                return state, completed_steps, True
            current_pose = state.privileged_state["end_effector_pose"]
            vla_input_state = self.primitive_policy.build_vla_input_state(primitive, current_pose)
            image = self.env.read_camera()
            cartesian_action = self.cartesian_policy.get_action(
                current_pose=current_pose,
                image=image,
                vla_input_state=vla_input_state,
                primitive_prompt=primitive.prompt,
                privileged_target_pose=primitive.target_pose,
            )

            self._draw_desired_path(
                current_pose,
                cartesian_action.cartesian_action_path,
                primitive_index,
            )

            next_state, reward = self.run_chunk(state, cartesian_action.cartesian_action_path)
            self._record_transition(
                completed_steps,
                state,
                reward,
                image,
                vla_input_state,
                primitive.prompt,
                primitive_index,
                cartesian_action,
            )

            state = next_state
            completed_steps += 1

            if self.training:
                self._sync_weights_if_due()

            if cartesian_action.completes_active_primitive:
                return state, completed_steps, False

    def _sync_weights_if_due(self):
        self.training_chunk_count += 1
        if self.training_chunk_count % self.cfg.training.sync_weights_every_n_chunks == 0:
            self._sync_weights()

    def _sync_weights(self):
        policy_weights = None
        try:
            while True:
                policy_weights = self.weights_queue.get_nowait()
        except queue.Empty:
            pass

        if policy_weights is not None:
            self.low_level_policy.policy.load_state_dict(policy_weights)

    def run_chunk(
        self,
        state: EnvironmentState,
        cartesian_action_path: np.ndarray,
    ) -> tuple[EnvironmentState, float]:
        """
        Executes a single chunk of low-level physics steps to chase the high-level action target.
        """
        start_positions = state.observation["joint_positions"].copy()

        chunk_start_pose_obj = state.privileged_state["end_effector_pose"]
        self.env.reset_chunk_reward_tracking(chunk_start_pose_obj, cartesian_action_path)
        policy_observation = self._build_policy_observation(state.observation, cartesian_action_path, start_positions, 0)

        total_reward = 0.0

        chunk_reward_metrics = {}

        for chunk_step in range(1, self.chunk_size + 1):
            chunk_terminated = self.cfg.training.terminate_at_chunk_end and chunk_step == self.chunk_size
            low_level_action, next_state, reward, reward_breakdown = self._step_low_level(
                policy_observation,
                cartesian_action_path,
                chunk_start_pose_obj,
                chunk_terminated,
            )

            total_reward += reward
            self._update_chunk_reward_metrics(chunk_reward_metrics, reward_breakdown)

            next_policy_observation = self._build_policy_observation(
                next_state.observation,
                cartesian_action_path,
                start_positions,
                chunk_step,
            )

            self._record_low_level_transition(
                policy_observation,
                next_policy_observation,
                low_level_action,
                reward,
                reward_breakdown,
                chunk_terminated,
                state,
                next_state,
            )
            self._add_to_replay_buffer(policy_observation, next_policy_observation, low_level_action, reward, chunk_terminated)

            self.episode_low_level_step += 1
            policy_observation = next_policy_observation

        self._publish_chunk_metrics(total_reward, chunk_reward_metrics)

        return next_state, total_reward
