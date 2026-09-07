import numpy as np
import queue
from omegaconf import DictConfig
from typing import Dict

from robot_arm.backends.servo import compensated_duty
from robot_arm.duty_compensation import DutyCompensator
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
        cartesian_hz = cfg.control.frequencies.cartesian
        joint_hz = cfg.control.frequencies.joint

        if joint_hz % cartesian_hz != 0:
            raise ValueError(f"joint_hz ({joint_hz}) must be divisible by cartesian_hz ({cartesian_hz})")

        self.env = env
        self.joint_steps_per_cartesian_action = joint_hz // cartesian_hz
        self.low_level_policy = low_level_policy
        self.primitive_policy = primitive_policy
        self.cartesian_policy = cartesian_policy
        self.training = training
        self.recorder = recorder
        self.replay_buffer = replay_buffer
        self.metrics_queue = metrics_queue
        self.cfg = cfg
        self.weights_queue = weights_queue
        self.joint_velocity_scale = cfg.control.joint_velocity_scale_radians_per_second
        self.duty_limits = np.array(
            [float(cfg.servo.max_duty[motor] / cfg.servo.full_scale_duty) for motor in self.env.motor_order],
            dtype=np.float32,
        )
        self.duty_compensator = DutyCompensator(self.env.arm.model, cfg.servo.stall_torque_newton_meters)

        self.cartesian_action_scale = waypoint_action_scale(
            cartesian_hz,
            cfg.waypoint.position_speed_meters_per_second,
            cfg.waypoint.rotation_speed_radians_per_second,
            cfg.waypoint.gripper_speed_radians_per_second,
        )

        self.tcp_velocity_scale = np.array(
            [cfg.waypoint.position_speed_meters_per_second] * 3 + [cfg.waypoint.rotation_speed_radians_per_second] * 3,
            dtype=np.float32,
        )

        self.max_cartesian_steps = int(cfg.control.max_seconds * cartesian_hz)
        self.episode_low_level_step = 0
        self.training_cartesian_action_count = 0

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
                box_gripped=state.box_gripped,
                state_idx=state_idx,
                primitive_index=primitive_index,
                obs=state.observation,
                sensor_state=state.sensor_state,
                pose=state.end_effector_pose,
                sim_state=state.sim_state if self.cfg.runtime.record_sim_state else None,
                images=self.env.read_cameras() if self.cfg.runtime.capture_camera else None,
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
        images: dict[str, np.ndarray],
        vla_input_state: np.ndarray,
        primitive_prompt: str,
        primitive_index: int,
        cartesian_action: CartesianAction,
    ):
        if self.recorder:
            self.recorder.record_transition(
                box_gripped=state.box_gripped,
                state_idx=state_idx,
                obs=state.observation,
                sensor_state=state.sensor_state,
                reward=reward,
                cartesian_action=cartesian_action.cartesian_action,
                pose=state.end_effector_pose,
                sim_state=state.sim_state if self.cfg.runtime.record_sim_state else None,
                images=images,
                vla_input_state=vla_input_state,
                primitive_prompt=primitive_prompt,
                primitive_index=primitive_index,
                diagnostics=cartesian_action.diagnostics,
                completes_active_primitive=cartesian_action.completes_active_primitive,
            )

    def _publish_cartesian_action_metrics(self, total_reward, cartesian_action_reward_metrics):
        if self.training and self.metrics_queue:
            detailed_metrics = {"total_reward": total_reward}
            detailed_metrics.update(self.env.arm.physics_metrics())

            if self.cfg.training.detailed_metrics:
                detailed_metrics.update(cartesian_action_reward_metrics)

            self.metrics_queue.add(detailed_metrics)

    def _build_policy_observation(
        self,
        observation: Dict[str, np.ndarray],
        remaining_delta: np.ndarray,
        joint_step: int,
        desired_gripper_duty: float,
        desired_gripper_duty_active: bool,
    ) -> Dict[str, np.ndarray]:
        policy_observation = dict(observation)
        policy_observation["joint_velocities"] = observation["joint_velocities"] / self.joint_velocity_scale
        policy_observation["tcp_velocity"] = observation["tcp_velocity"] / self.tcp_velocity_scale
        policy_observation["remaining_delta"] = remaining_delta / self.cartesian_action_scale
        policy_observation["time_left"] = np.array(
            [(self.joint_steps_per_cartesian_action - joint_step) / self.joint_steps_per_cartesian_action],
            dtype=np.float32,
        )
        policy_observation["desired_gripper_duty"] = np.array([desired_gripper_duty], dtype=np.float32)
        policy_observation["desired_gripper_duty_active"] = np.array([float(desired_gripper_duty_active)], dtype=np.float32)
        policy_observation["gripper_duty_difference"] = np.array(
            [desired_gripper_duty - float(observation["gripper_duty"][0])],
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
        cartesian_action_terminated: bool,
        duty_compensation: np.ndarray,
        compensated_duty_action: np.ndarray,
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
                cartesian_action_terminated,
                duty_compensation,
                compensated_duty_action,
                state,
                next_state,
            )

    def _add_to_replay_buffer(
        self,
        policy_observation: Dict[str, np.ndarray],
        next_policy_observation: Dict[str, np.ndarray],
        low_level_action: np.ndarray,
        reward: float,
        cartesian_action_terminated: bool,
    ):
        if self.training and self.replay_buffer:
            self.replay_buffer.add(
                policy_observation,
                next_policy_observation,
                low_level_action,
                reward,
                cartesian_action_terminated,
            )

    def _step_low_level(
        self,
        policy_observation: Dict[str, np.ndarray],
        cartesian_action: np.ndarray,
        cartesian_action_start_pose: object,
        cartesian_action_ends: bool,
        desired_gripper_duty: float,
        desired_gripper_duty_active: bool,
    ):
        low_level_action, _ = self.low_level_policy.predict(policy_observation, deterministic=not self.training)
        duty_compensation = self.duty_compensator.calculate(
            policy_observation["joint_positions"],
            policy_observation["joint_velocities"] * self.joint_velocity_scale,
        )
        compensated_duty_action = compensated_duty(low_level_action, duty_compensation, self.duty_limits)
        next_state, reward, reward_breakdown = self.env.step(
            compensated_duty_action,
            policy_observation["joint_positions"],
            low_level_action,
            float(policy_observation["time_left"][0]),
            cartesian_action,
            cartesian_action_start_pose,
            cartesian_action_ends,
            desired_gripper_duty,
            desired_gripper_duty_active,
        )

        if self.cfg.runtime.draw_tcp:
            self.env.arm.draw_tcp()

        return low_level_action, duty_compensation, compensated_duty_action, next_state, reward, reward_breakdown

    def _update_cartesian_action_reward_metrics(self, cartesian_action_reward_metrics, reward_breakdown):
        if self.cfg.training.detailed_metrics:
            for key, value in reward_breakdown.items():
                if key not in cartesian_action_reward_metrics:
                    cartesian_action_reward_metrics[key] = []
                cartesian_action_reward_metrics[key].append(value)
            if self.cfg.training.pose_delta_diagnostics_enabled:
                for key, value in self.env.pose_delta_diagnostics.items():
                    if key not in cartesian_action_reward_metrics:
                        cartesian_action_reward_metrics[key] = []
                    cartesian_action_reward_metrics[key].append(value)

    def run_episode(self, generate_primitives: bool):
        try:
            self._run_episode(generate_primitives, self.env.reset())
        finally:
            if self.recorder:
                self.recorder.save()

    def run_episode_from_sim_state(self, generate_primitives: bool, qpos: np.ndarray, qvel: np.ndarray):
        try:
            self._run_episode(generate_primitives, self.env.reset_from_sim_state(qpos, qvel))
        finally:
            if self.recorder:
                self.recorder.save()

    def _run_episode(self, generate_primitives: bool, state: EnvironmentState):
        self._prepare_primitives(generate_primitives, start_pose=state.end_effector_pose)
        self._initialize_debug_visualization()

        completed_steps = 0
        while self.primitive_policy.has_next_primitive():
            primitive_index, primitive = self.primitive_policy.get_next_primitive(
                state.end_effector_pose,
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
            if completed_steps >= self.max_cartesian_steps:
                return state, completed_steps, True
            current_pose = state.end_effector_pose
            gripper_duty = float(state.observation["gripper_duty"][0])
            vla_input_state = self.primitive_policy.build_vla_input_state(primitive, current_pose, gripper_duty)
            images = self.env.read_cameras() if self.cfg.runtime.capture_camera else None
            cartesian_action = self.cartesian_policy.get_action(
                current_pose=current_pose,
                images=images,
                vla_input_state=vla_input_state,
                gripper_duty=gripper_duty,
                primitive=primitive,
            )

            self._draw_desired_path(
                current_pose,
                cartesian_action.cartesian_action,
                primitive_index,
            )

            next_state, reward = self.execute_cartesian_action(
                state,
                cartesian_action.cartesian_action,
                primitive.desired_gripper_duty,
                primitive.desired_gripper_duty_active,
            )
            self._record_transition(
                completed_steps,
                state,
                reward,
                images,
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
        self.training_cartesian_action_count += 1
        if self.training_cartesian_action_count % self.cfg.training.sync_weights_every_n_cartesian_actions == 0:
            self._sync_weights()

    def _sync_weights(self):
        policy_weights = None
        try:
            while True:
                policy_weights = self.weights_queue.get_nowait()
        except queue.Empty:
            pass

        if policy_weights is not None:
            self.low_level_policy.set_actor_params(policy_weights)

    def execute_cartesian_action(
        self,
        state: EnvironmentState,
        cartesian_action: np.ndarray,
        desired_gripper_duty: float,
        desired_gripper_duty_active: bool,
    ) -> tuple[EnvironmentState, float]:
        """
        Executes the joint steps of one cartesian action, all chasing the pose that action asks for.
        """
        cartesian_action_start_pose_obj = state.end_effector_pose
        desired_pose = cartesian_action_start_pose_obj.apply_delta(cartesian_action)
        self.env.reset_cartesian_action_reward_tracking(cartesian_action_start_pose_obj, cartesian_action)
        policy_observation = self._build_policy_observation(
            state.observation,
            cartesian_action_start_pose_obj.delta_to(desired_pose),
            0,
            desired_gripper_duty,
            desired_gripper_duty_active,
        )

        total_reward = 0.0

        cartesian_action_reward_metrics = {}

        for joint_step in range(1, self.joint_steps_per_cartesian_action + 1):
            cartesian_action_ends = joint_step == self.joint_steps_per_cartesian_action
            cartesian_action_terminated = self.cfg.training.terminate_at_cartesian_action_end and cartesian_action_ends
            low_level_action, duty_compensation, compensated_duty_action, next_state, reward, reward_breakdown = self._step_low_level(
                policy_observation,
                cartesian_action,
                cartesian_action_start_pose_obj,
                cartesian_action_ends,
                desired_gripper_duty,
                desired_gripper_duty_active,
            )

            total_reward += reward
            self._update_cartesian_action_reward_metrics(cartesian_action_reward_metrics, reward_breakdown)

            next_policy_observation = self._build_policy_observation(
                next_state.observation,
                next_state.end_effector_pose.delta_to(desired_pose),
                joint_step,
                desired_gripper_duty,
                desired_gripper_duty_active,
            )

            self._record_low_level_transition(
                policy_observation,
                next_policy_observation,
                low_level_action,
                reward,
                reward_breakdown,
                cartesian_action_terminated,
                duty_compensation,
                compensated_duty_action,
                state,
                next_state,
            )
            self._add_to_replay_buffer(policy_observation, next_policy_observation, low_level_action, reward, cartesian_action_terminated)

            self.episode_low_level_step += 1
            policy_observation = next_policy_observation

        self._publish_cartesian_action_metrics(total_reward, cartesian_action_reward_metrics)

        return next_state, total_reward
