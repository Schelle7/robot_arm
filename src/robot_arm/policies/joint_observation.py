from typing import Dict

import numpy as np

from robot_arm.control_types import CartesianAction, EnvironmentState
from robot_arm.geometry.pose import Pose
from robot_arm.policies.cartesian import waypoint_action_scale
from robot_arm.robot_schema import HISTORY_JOINT_VELOCITY_SLICE, HISTORY_TCP_VELOCITY_SLICE, policy_observation_sizes


class JointObservationBuilder:
    def __init__(self, cfg, history_steps: int):
        cartesian_hz = cfg.control.frequencies.cartesian
        self.joint_velocity_scale = cfg.control.joint_velocity_scale_radians_per_second
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
        self.policy_observation_sizes = policy_observation_sizes(
            int(cfg.waypoint.cartesian_action_dim),
            history_steps,
        )

    def build(
        self,
        state: EnvironmentState,
        cartesian_action: CartesianAction,
        desired_pose: Pose,
        cartesian_action_progress: float,
    ) -> Dict[str, np.ndarray]:
        observation = state.observation
        remaining_delta = state.end_effector_pose.delta_to(desired_pose)
        history_steps = observation["policy_history"].copy()
        history_steps[:, HISTORY_JOINT_VELOCITY_SLICE] /= self.joint_velocity_scale
        history_steps[:, HISTORY_TCP_VELOCITY_SLICE] /= self.tcp_velocity_scale
        history = np.concatenate((observation["joint_positions"], history_steps.reshape(-1))).astype(np.float32)

        policy_state = np.concatenate(
            (
                observation["joint_positions"],
                observation["joint_velocities"] / self.joint_velocity_scale,
                observation["tcp_position"],
                observation["tcp_velocity"] / self.tcp_velocity_scale,
                observation["gripper_duty"],
            )
        ).astype(np.float32)
        goal = np.concatenate(
            (
                remaining_delta / self.cartesian_action_scale,
                np.array(
                    [
                        1.0 - cartesian_action_progress,
                        cartesian_action.desired_gripper_duty,
                        float(cartesian_action.desired_gripper_duty_active),
                        cartesian_action.desired_gripper_duty - float(observation["gripper_duty"][0]),
                    ],
                    dtype=np.float32,
                ),
            )
        )
        policy_observation = {"history": history, "state": policy_state, "goal": goal}
        for name, values in policy_observation.items():
            assert values.shape == (self.policy_observation_sizes[name],)
        return policy_observation
