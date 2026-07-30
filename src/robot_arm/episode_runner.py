import numpy as np
from omegaconf import DictConfig
from typing import Dict

from robot_arm.policies import Policy
from robot_arm.envs.env import RobotEnv
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
        high_level_policy: Policy,
        training: bool,
        recorder: EpisodeRecorder,
        replay_buffer=None,
    ):
        high_level_hz = cfg.frequencies.high_level
        low_level_hz = cfg.frequencies.low_level

        if low_level_hz % high_level_hz != 0:
            raise ValueError(
                f"low_level_hz ({low_level_hz}) must be divisible by high_level_hz ({high_level_hz})"
            )

        self.env = env
        self.chunk_size = low_level_hz // high_level_hz
        self.low_level_policy = low_level_policy
        self.high_level_policy = high_level_policy
        self.training = training
        self.recorder = recorder
        self.replay_buffer = replay_buffer
        self.cfg = cfg

        self.max_high_level_steps = int(cfg.max_seconds * high_level_hz)
        self.episode_low_level_step = 0

    def run_episode(
        self, instruction: str, generate_waypoints: bool, **waypoint_kwargs
    ):
        obs = self.env.reset()

        if generate_waypoints and hasattr(
            self.high_level_policy, "generate_grab_waypoints"
        ):
            waypoint_kwargs["box_pose"] = self.env.arm.get_privileged_box_pose()
            self.high_level_policy.generate_grab_waypoints(**waypoint_kwargs)

        if self.cfg.draw_waypoints:
            self.env.arm.draw_waypoints(self.high_level_policy.waypoints)
            self.recorder.save_waypoints(self.high_level_policy.waypoints)

        try:
            for step_idx in range(self.max_high_level_steps):
                high_level_action = self.high_level_policy.get_action(
                    obs,
                    privileged_end_effector_pose=self.env.get_privileged_end_effector_pose(),
                    instruction=instruction,
                )

                next_obs, reward = self.run_chunk(obs, high_level_action)

                if self.recorder:
                    image = self.env.read_camera()
                    pose_ee, gripper_state = self.env.get_privileged_end_effector_pose()
                    
                    info_dict = {
                        "image": image, 
                        "privileged_end_effector_pose": pose_ee,
                        "gripper_state": gripper_state,
                        "high_level_action": high_level_action,
                    }
                    
                    if self.cfg["record_sim_state"]:
                        info_dict["sim_state"] = self.env.arm.read_state()["sim_state"]

                    self.recorder.record_high_level(
                        step_idx,
                        obs,
                        reward=reward,
                        info=info_dict,
                        instruction=instruction,
                    )

                obs = next_obs

        except BaseException as e:
            print(f"\nExecution interrupted by {type(e).__name__}: {e}")
            raise
        finally:
            if self.recorder:
                self.recorder.save()

    def run_chunk(
        self, raw_obs: Dict[str, np.ndarray], high_level_action: np.ndarray
    ) -> tuple[Dict[str, np.ndarray], float]:
        """
        Executes a single chunk of low-level physics steps to chase the high-level action target.
        """
        start_positions = raw_obs["joint_positions"].copy()

        # We need the physical cartesian start pose explicitly to calculate rewards
        # The env doesn't track this statefully anymore, we provide it.
        # It's hidden in the observation info block from either reset() or the last chunk loop.
        # Since _get_obs doesn't return info directly to run_chunk via arg, we get it here.
        chunk_start_pose_obj, _ = self.env.get_privileged_end_effector_pose()

        # Construct the first policy observation before entering the loop
        policy_obs = dict(raw_obs)
        policy_obs["high_level_action"] = high_level_action
        policy_obs["start_joint_positions"] = start_positions
        policy_obs["time_left"] = np.array([self.chunk_size], dtype=np.float32)

        total_reward = 0.0

        for chunk_step in range(self.chunk_size):
            low_level_action, _ = self.low_level_policy.predict(
                policy_obs, deterministic=not self.training
            )

            chunk_terminated = chunk_step == self.chunk_size - 1

            # Env returns physical state (raw_next_obs) plus step metrics
            raw_next_obs, reward = self.env.step(
                low_level_action, high_level_action, chunk_start_pose_obj, chunk_terminated
            )

            total_reward += reward

            # Construct the next policy observation for the RL transition and next step
            next_policy_obs = dict(raw_next_obs)
            next_policy_obs["high_level_action"] = high_level_action
            next_policy_obs["start_joint_positions"] = start_positions
            next_policy_obs["time_left"] = np.array(
                [self.chunk_size - chunk_step - 1], dtype=np.float32
            )

            chunk_terminated = chunk_step == self.chunk_size - 1

            if self.recorder:
                self.recorder.append_low_level_transition(
                    policy_obs,
                    next_policy_obs,
                    low_level_action,
                    reward,
                    chunk_terminated,
                )

            if self.training and self.replay_buffer:
                self.replay_buffer.add(
                    policy_obs,
                    next_policy_obs,
                    low_level_action,
                    reward,
                    chunk_terminated,
                )

            self.episode_low_level_step += 1
            policy_obs = next_policy_obs

        # Return physical state for the next high-level policy inference
        return raw_next_obs, total_reward
