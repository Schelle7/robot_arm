import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List

import hydra
import mujoco
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from robot_arm.envs.factory import make_env
from robot_arm.envs.safety import SafetyException
from robot_arm.recording.model_snapshot import snapshot_model_files
from robot_arm.recording.recorder import EpisodeRecorder
from robot_arm.robot_schema import BOX_BODY_NAMES, CARTESIAN_ACTION_NAMES, MOTOR_ORDER


@dataclass
class DeltaResult:
    commanded_delta_radians: float
    gripper_radians: float
    duty_fraction: float
    smoothed_duty: float
    tripped_safety: bool


def move_box(arm, body_name: str, position) -> None:
    body_id = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    joint_id = arm.model.body_jntadr[body_id]
    qpos_adr = arm.model.jnt_qposadr[joint_id]
    dof_adr = arm.model.jnt_dofadr[joint_id]
    arm.data.qpos[qpos_adr : qpos_adr + 3] = position
    arm.data.qpos[qpos_adr + 3 : qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
    arm.data.qvel[dof_adr : dof_adr + 6] = 0.0


def place_measurement_scene(arm, start_joint_positions) -> None:
    """
    Poses the arm around a box resting on the floor, free to shift and slip the way a real one
    would. The real backend reaches the same geometry by being posed by hand before power-on.
    """
    for name, radians in start_joint_positions.items():
        arm.data.qpos[arm.model.jnt_qposadr[arm.joint_indices[name]]] = radians
        arm.data.ctrl[arm.actuator_indices[name]] = radians
    mujoco.mj_forward(arm.model, arm.data)

    box_body_id = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_BODY, BOX_BODY_NAMES[0])
    resting_height = float(arm.model.geom_size[arm.model.body_geomadr[box_body_id]][2])
    jaw_midpoint = arm.get_tcp_pose(arm.read_state()).position
    move_box(arm, BOX_BODY_NAMES[0], (jaw_midpoint[0], jaw_midpoint[1], resting_height))
    mujoco.mj_forward(arm.model, arm.data)


def joint_targets(state: Dict, gripper_target_radians: float) -> Dict[str, float]:
    """Every joint at its measured value so only the gripper is commanded anywhere new."""
    positions = {motor: state["Present_Position"][motor] for motor in MOTOR_ORDER}
    positions["gripper"] = gripper_target_radians
    return positions


class GripperSweep:
    def __init__(self, cfg: DictConfig, env, recorder: EpisodeRecorder):
        self.cfg = cfg
        self.env = env
        self.arm = env.arm
        self.recorder = recorder
        self.is_sim = cfg.backend == "sim"
        self.recorded_states = 0
        self.held_poses: List[np.ndarray] = []
        self.last_state = self.arm.read_state()
        self.chunk_size = cfg.control.frequencies.joint // cfg.control.frequencies.cartesian
        assert cfg.sweep.hold_steps % self.chunk_size == 0, "hold_steps must be a whole number of mid-level chunks."

    def command_gripper(self, state: Dict, gripper_target_radians: float) -> float:
        """
        Issues one low-level step and returns the normalized action a policy would have had to emit
        for the same joint delta, so the recording is comparable to a real rollout.
        """
        self.arm.write_goal(joint_targets(state, gripper_target_radians))
        self.arm.advance_control_step()
        return (gripper_target_radians - state["Present_Position"]["gripper"]) / self.env.delta_action_scale

    def observation_of(self, state: Dict) -> Dict[str, np.ndarray]:
        return {
            "joint_positions": np.array([state["Present_Position"][m] for m in MOTOR_ORDER], dtype=np.float32),
            "joint_velocities": np.array([state["Present_Velocity"][m] for m in MOTOR_ORDER], dtype=np.float32),
        }

    def append_low_level(self, state: Dict, next_state: Dict, gripper_action: float) -> None:
        action = np.zeros(len(MOTOR_ORDER), dtype=np.float32)
        action[MOTOR_ORDER.index("gripper")] = gripper_action
        self.recorder.append_low_level_transition(
            self.observation_of(state),
            self.observation_of(next_state),
            action,
            0.0,
            {"duty_fraction": next_state["Present_Load"]["gripper"]},
            False,
            np.zeros(len(MOTOR_ORDER), dtype=np.float32),
            action,
            SimpleNamespace(end_effector_pose=self.arm.get_tcp_pose(state)),
            SimpleNamespace(end_effector_pose=self.arm.get_tcp_pose(next_state)),
        )

    def record(self, state: Dict, delta_index: int, commanded_delta_radians: float, duty_fraction: float) -> None:
        pose = self.arm.get_tcp_pose(state)
        self.recorder.record_transition(
            grasp_confirmed=self.env.grasp_estimator.update(
                state["Present_Position"]["gripper"],
                state["Present_Velocity"]["gripper"],
                state["Present_Load"]["gripper"],
                state["sample_time_ns"],
            ),
            state_idx=self.recorded_states,
            obs=self.observation_of(state),
            sensor_state=state,
            reward=0.0,
            cartesian_action_path=np.zeros((1, len(CARTESIAN_ACTION_NAMES)), dtype=np.float32),
            pose=pose,
            sim_state=state["sim_state"] if self.cfg.runtime.record_sim_state and self.is_sim else None,
            images=self.env.read_cameras(),
            vla_input_state=np.zeros(16, dtype=np.float32),
            primitive_prompt=f"hold gripper {commanded_delta_radians:.4f} rad below measured",
            primitive_index=delta_index,
            diagnostics={
                "commanded_delta_radians": commanded_delta_radians,
                "duty_fraction": duty_fraction,
                "gripper_radians": float(state["Present_Position"]["gripper"]),
            },
            completes_active_primitive=False,
        )
        self.recorded_states += 1

    def release(self) -> None:
        """
        Opens at a bounded rate. Commanding the open angle outright is a position error large enough
        to saturate the actuator on its own, which drives the safety load EMA up before any squeeze.
        """
        step_radians = float(self.cfg.sweep.release_step_radians)
        for _ in range(self.cfg.sweep.release_steps):
            state = self.arm.read_state()
            self.last_state = state
            current = state["Present_Position"]["gripper"]
            target = current + float(np.clip(self.cfg.sweep.open_radians - current, -step_radians, step_radians))
            self.command_gripper(state, target)

        # Held still so the load EMA decays. Opening costs damping force, so without this every
        # squeeze would start from the load the release left behind rather than from rest.
        for _ in range(self.cfg.sweep.idle_steps):
            state = self.arm.read_state()
            self.last_state = state
            self.command_gripper(state, state["Present_Position"]["gripper"])

    def measure(self, delta_index: int, commanded_delta_radians: float) -> DeltaResult:
        state = self.arm.read_state()
        chunk_start_state = state
        for _ in range(self.cfg.sweep.hold_steps):
            gripper_action = self.command_gripper(state, state["Present_Position"]["gripper"] - commanded_delta_radians)
            next_state = self.arm.read_state()
            self.last_state = next_state
            self.append_low_level(state, next_state, gripper_action)
            state = next_state

            # One frame with an image per mid-level chunk, with every low-level step nested inside it.
            if len(self.recorder.dense_trajectory_buffer) == self.chunk_size:
                self.record(chunk_start_state, delta_index, commanded_delta_radians, state["Present_Load"]["gripper"])
                chunk_start_state = state

        self.held_poses.append(self.arm.get_tcp_pose(state).as_10d())
        return DeltaResult(
            commanded_delta_radians=commanded_delta_radians,
            gripper_radians=float(state["Present_Position"]["gripper"]),
            duty_fraction=float(state["Present_Load"]["gripper"]),
            smoothed_duty=float(self.arm.smoothed_duties["gripper"]),
            tripped_safety=False,
        )

    def run(self) -> List[DeltaResult]:
        results = []
        for delta_index, commanded_delta_radians in enumerate(self.cfg.sweep.commanded_deltas_radians):
            try:
                self.release()
                results.append(self.measure(delta_index, float(commanded_delta_radians)))
            except SafetyException as safety_stop:
                # The wrapper torque-offs the bus, so the sweep cannot continue past its first trip.
                print(f"\nSafety stop at delta {commanded_delta_radians}: {safety_stop}")
                results.append(
                    DeltaResult(
                        commanded_delta_radians=float(commanded_delta_radians),
                        gripper_radians=float("nan"),
                        duty_fraction=float("nan"),
                        smoothed_duty=float("nan"),
                        tripped_safety=True,
                    )
                )
                break

        self.recorder.save_waypoints(self.held_poses)
        self.record_final_state()
        return results

    def record_final_state(self) -> None:
        # Reuses the last reading rather than taking a fresh one, because after a safety stop the
        # EMA is already over the limit and every further read would raise again.
        state = self.last_state
        self.recorder.record_final_state(
            grasp_confirmed=self.env.grasp_estimator.update(
                self.last_state["Present_Position"]["gripper"],
                self.last_state["Present_Velocity"]["gripper"],
                self.last_state["Present_Load"]["gripper"],
                self.last_state["sample_time_ns"],
            ),
            state_idx=self.recorded_states,
            primitive_index=max(len(self.held_poses) - 1, 0),
            obs={
                "joint_positions": np.array([state["Present_Position"][m] for m in MOTOR_ORDER], dtype=np.float32),
                "joint_velocities": np.array([state["Present_Velocity"][m] for m in MOTOR_ORDER], dtype=np.float32),
            },
            sensor_state=state,
            pose=self.arm.get_tcp_pose(state),
            sim_state=state["sim_state"] if self.cfg.runtime.record_sim_state and self.is_sim else None,
            images=self.env.read_cameras(),
        )


def print_results(results: List[DeltaResult], max_smoothed_duty: float) -> None:
    print(f"\n{'delta rad':>12} {'gripper rad':>12} {'load frac':>11} {'smoothed':>10}  outcome")
    for result in results:
        outcome = f"SAFETY STOP (limit {max_smoothed_duty})" if result.tripped_safety else "held"
        print(f"{result.commanded_delta_radians:12.4f} {result.gripper_radians:12.4f} " f"{result.duty_fraction:11.4f} {result.smoothed_duty:10.4f}  {outcome}")


@hydra.main(version_base=None, config_path="../conf", config_name="characterize_gripper")
def main(cfg: DictConfig):
    run_dir = HydraConfig.get().runtime.output_dir
    snapshot_model_files(cfg.model_path, run_dir)

    env = make_env(cfg, run_dir)
    env.reset(enable_added_weight=False)
    if cfg.backend == "sim":
        place_measurement_scene(env.arm, cfg.sweep.start_joint_positions)

    recorder = EpisodeRecorder(
        output_dir=os.path.join(run_dir, "recordings"),
        cfg=cfg,
        episode_name="gripper_sweep",
    )

    sweep = GripperSweep(cfg, env, recorder)
    for remaining in range(int(cfg.sweep.countdown_seconds), 0, -1):
        print(f"Gripping in {remaining}...", flush=True)
        time.sleep(1.0)
    print("Go.", flush=True)

    try:
        results = sweep.run()
    finally:
        recorder.save()

    print_results(results, cfg.safety.max_smoothed_duty)
    print(f"\nReplay with:\n  python scripts/replay.py episode_path={os.path.join(run_dir, 'recordings', 'gripper_sweep', 'episode.npz')}")


if __name__ == "__main__":
    main()
