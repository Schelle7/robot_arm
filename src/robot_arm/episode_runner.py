import numpy as np
import queue
from omegaconf import DictConfig
from typing import Dict

from robot_arm.policies.cartesian import CartesianPolicy, ScriptedCartesianPolicy
from robot_arm.policies.primitive_generator import ScriptedPrimitiveGeneratorPolicy
from robot_arm.envs.env import RobotEnv
from robot_arm.control_types import ActionPrimitive, CartesianAction, EnvironmentState
from robot_arm.recording.recorder import EpisodeRecorder
from robot_arm.policies.joint_observation import JointObservationBuilder


class EpisodeRunner:
    """
    Orchestrates the entire episode.
    Handles Cartesian/joint synchronization, logging to recorder,
    and returns transitions for the training buffers.
    """

    def __init__(
        self,
        cfg: DictConfig,
        env: RobotEnv,
        joint_policy,
        primitive_policy: ScriptedPrimitiveGeneratorPolicy,
        cartesian_policy: CartesianPolicy,
        training: bool,
        recorder: EpisodeRecorder,
        replay_buffer,
        metrics_queue,
        weights_queue,
        progress,
    ):
        cartesian_hz = cfg.control.frequencies.cartesian
        joint_hz = cfg.control.frequencies.joint

        if joint_hz % cartesian_hz != 0:
            raise ValueError(f"joint_hz ({joint_hz}) must be divisible by cartesian_hz ({cartesian_hz})")

        abort_angle = float(cfg.waypoint.pick_and_place.gripper_abort_below_radians)
        closed_angle = float(cfg.waypoint.pick_and_place.gripper_closed_radians)
        if abort_angle < closed_angle:
            raise ValueError(f"Gripper abort threshold ({abort_angle} rad) must not be below the requested closed angle ({closed_angle} rad).")

        self.env = env
        self.joint_steps_per_cartesian_action = joint_hz // cartesian_hz
        self.joint_policy = joint_policy
        self.primitive_policy = primitive_policy
        self.cartesian_policy = cartesian_policy
        self.teacher_policy = ScriptedCartesianPolicy(cfg)
        self.training = training
        self.recorder = recorder
        self.replay_buffer = replay_buffer
        self.metrics_queue = metrics_queue
        self.cfg = cfg
        self.weights_queue = weights_queue
        self.progress = progress
        self.duty_limits = np.array(
            [float(cfg.servo.max_duty[motor] / cfg.servo.full_scale_duty) for motor in self.env.motor_order],
            dtype=np.float32,
        )

        self.joint_observation = JointObservationBuilder(cfg, self.env.policy_history_steps)

        self.max_cartesian_steps = int(cfg.control.max_seconds * cartesian_hz)
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
                state=state,
                state_idx=state_idx,
                primitive_index=primitive_index,
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
        primitive: ActionPrimitive,
        primitive_index: int,
        cartesian_action: CartesianAction,
    ):
        if self.recorder:
            teacher_action = self.teacher_policy.get_action(
                state=state,
                images=images,
                vla_input_state=vla_input_state,
                primitive=primitive,
            )
            self.recorder.record_transition(
                state_idx=state_idx,
                state=state,
                reward=reward,
                cartesian_action=cartesian_action,
                teacher_action=teacher_action,
                images=images,
                vla_input_state=vla_input_state,
                primitive_prompt=primitive.prompt,
                primitive_index=primitive_index,
            )

    def _publish_cartesian_action_metrics(self, total_reward, cartesian_action_reward_metrics):
        if self.training and self.metrics_queue:
            detailed_metrics = {"total_reward": total_reward}
            detailed_metrics.update(self.env.arm.physics_metrics())

            if self.cfg.training.detailed_metrics:
                detailed_metrics.update(cartesian_action_reward_metrics)

            self.metrics_queue.add(detailed_metrics)

    def _record_joint_transition(
        self,
        policy_observation: Dict[str, np.ndarray],
        next_policy_observation: Dict[str, np.ndarray],
        joint_action: np.ndarray,
        reward: float,
        reward_breakdown: Dict[str, float],
        cartesian_action_terminated: bool,
        duty_action: np.ndarray,
        state: EnvironmentState,
        next_state: EnvironmentState,
    ):
        if self.recorder and self.cfg.runtime.record_policy_debug:
            self.recorder.append_joint_transition(
                policy_observation,
                next_policy_observation,
                joint_action,
                reward,
                reward_breakdown,
                cartesian_action_terminated,
                duty_action,
                state,
                next_state,
            )

    def _add_to_replay_buffer(
        self,
        policy_observation: Dict[str, np.ndarray],
        next_policy_observation: Dict[str, np.ndarray],
        joint_action: np.ndarray,
        reward: float,
        cartesian_action_terminated: bool,
    ):
        if self.training and self.replay_buffer:
            self.replay_buffer.add(
                policy_observation,
                next_policy_observation,
                joint_action,
                reward,
                cartesian_action_terminated,
            )

    def _step_joint(
        self,
        policy_observation: Dict[str, np.ndarray],
        state: EnvironmentState,
        joint_step_idx: int,
        cartesian_action: CartesianAction,
        cartesian_action_start_pose: object,
    ):
        joint_action, _ = self.joint_policy.predict(policy_observation, deterministic=not self.training)
        duty_action = joint_action * self.duty_limits
        next_state, reward, reward_breakdown = self.env.step(
            duty_action,
            state,
            joint_action,
            joint_step_idx,
            cartesian_action,
            cartesian_action_start_pose,
        )

        if self.cfg.runtime.draw_tcp:
            self.env.arm.draw_tcp()

        return joint_action, duty_action, next_state, reward, reward_breakdown

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
            if generate_primitives:
                self.primitive_policy.select_task()
            state = self.env.reset(enable_added_weight=self.primitive_policy.task != "pick_and_place")
            self._run_episode(generate_primitives, state)
        finally:
            if self.recorder:
                self.recorder.save()

    def run_episode_from_sim_state(self, generate_primitives: bool, qpos: np.ndarray, qvel: np.ndarray):
        try:
            if generate_primitives:
                self.primitive_policy.select_task()
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
        if self.progress is not None:
            self.progress.set_description(primitive.prompt)

        while completed_steps < self.max_cartesian_steps:
            vla_input_state = self.primitive_policy.build_vla_input_state(primitive, state)
            images = self.env.read_cameras() if self.cfg.runtime.capture_camera else None

            cartesian_action = self.cartesian_policy.get_action(
                state=state,
                images=images,
                vla_input_state=vla_input_state,
                primitive=primitive,
            )

            self._draw_desired_path(
                state.end_effector_pose,
                cartesian_action.cartesian_action,
                primitive_index,
            )

            next_state, reward = self.execute_cartesian_action(
                state,
                cartesian_action,
            )
            self._record_transition(
                completed_steps,
                state,
                reward,
                images,
                vla_input_state,
                primitive,
                primitive_index,
                cartesian_action,
            )

            state = next_state
            completed_steps += 1
            if self.progress is not None:
                self.progress.update(1)

            if self.training:
                self._sync_weights_if_due()

            if self.grip_failed(state, primitive):
                return state, completed_steps, True

            if cartesian_action.completes_active_primitive:
                return state, completed_steps, False

        return state, completed_steps, True

    def grip_failed(self, state: EnvironmentState, primitive: ActionPrimitive) -> bool:
        return bool(
            primitive.desired_gripper_duty_active
            and (state.end_effector_pose.gripper < self.cfg.waypoint.pick_and_place.gripper_abort_below_radians or self.env.box_too_far(state.end_effector_pose))
        )

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
            self.joint_policy.set_actor_params(policy_weights)

    def execute_cartesian_action(
        self,
        state: EnvironmentState,
        cartesian_action: CartesianAction,
    ) -> tuple[EnvironmentState, float]:
        """
        Executes the joint steps of one cartesian action, all chasing the pose that action asks for.
        """
        cartesian_action_start_pose_obj = state.end_effector_pose
        desired_pose = cartesian_action_start_pose_obj.apply_delta(cartesian_action.cartesian_action)
        self.env.reset_cartesian_action_reward_tracking(cartesian_action_start_pose_obj, cartesian_action.cartesian_action)
        cartesian_action_progress = 0.0
        policy_observation = self.joint_observation.build(
            state,
            cartesian_action,
            desired_pose,
            cartesian_action_progress,
        )

        total_reward = 0.0

        cartesian_action_reward_metrics = {}

        for joint_step_idx in range(1, self.joint_steps_per_cartesian_action + 1):
            cartesian_action_ends = joint_step_idx == self.joint_steps_per_cartesian_action
            cartesian_action_terminated = self.cfg.training.terminate_at_cartesian_action_end and cartesian_action_ends
            joint_action, duty_action, next_state, reward, reward_breakdown = self._step_joint(
                policy_observation,
                state,
                joint_step_idx,
                cartesian_action,
                cartesian_action_start_pose_obj,
            )

            total_reward += reward
            self._update_cartesian_action_reward_metrics(cartesian_action_reward_metrics, reward_breakdown)

            cartesian_action_progress = joint_step_idx / self.joint_steps_per_cartesian_action
            next_policy_observation = self.joint_observation.build(
                next_state,
                cartesian_action,
                desired_pose,
                cartesian_action_progress,
            )

            self._record_joint_transition(
                policy_observation,
                next_policy_observation,
                joint_action,
                reward,
                reward_breakdown,
                cartesian_action_terminated,
                duty_action,
                state,
                next_state,
            )
            self._add_to_replay_buffer(policy_observation, next_policy_observation, joint_action, reward, cartesian_action_terminated)

            policy_observation = next_policy_observation
            state = next_state

        self._publish_cartesian_action_metrics(total_reward, cartesian_action_reward_metrics)

        return next_state, total_reward
