# Development notes

Implementation details and development workflows. For an overview and common
commands, see [README.md](README.md). Instructions for coding agents belong in
`AGENTS.md`, separately from these technical notes.

## Control stack

1. SmolVLA maps camera images, prompts, and proprioception to Cartesian pose deltas and primitive completion.
2. SAC maps Cartesian tracking state to joint-duty actions centered on model-derived compensation.
3. The backend applies duties in MuJoCo or communicates with the physical SO-101.

Pick-and-place uses separate open, move above box, descend, close, lift, transport, lower, and release primitives. Visual primitives retain privileged targets for supervision but do not expose target offsets to SmolVLA.

TCP position is the fingertip midpoint and its primary axis points from the moving fingertip toward the fixed fingertip. `gripper_geometry.py` computes geometry at the requested opening in a separate MuJoCo data object, with the fixed fingertip as the local reference. `waypoints.py` uses that geometry to adjust radial placement and rotate the whole assembly around the shoulder pivot until the TCP matches the target, then takes the resulting orientation. This uses the fingertip reference approximation and does not check full-arm reachability or collisions.

The Cartesian runner aborts closing and holding phases below 0.30 rad gripper position after each action. Primitive completion is decided by the Cartesian policy. The recorded grasp flag uses a 0.2-second continuous hold of the configured angle, duty, and velocity conditions. In simulation, the box center must also stay within 4 cm of the TCP. Training defaults specify a 30-second episode limit, with objects placed 25 to 35 cm horizontally from the shoulder pivot.

## Backend status

- Simulation supports duty-driven motor dynamics and state replay.
- Hardware position, velocity, load, temperature, voltage, and current reads are available.
- Real-arm direct PWM duty output is implemented.

## Training and data workflows

SAC training includes independent forward dynamics heads for the actor and both critics.
Each predicts six joint and six TCP interval velocities from its own history and state encodings
plus the recorded action, using the observation's existing velocity normalization. Predictions
are auxiliary training outputs, not inputs to the Q-value or action outputs. Configure hidden
layers with `policy.forward_head` and loss weights with `training.actor_forward_loss_weight`
and `training.critic_forward_loss_weight` (initially 0.1 each; zero removes the respective gradient
contribution). TensorBoard reports `actor_loss`, `actor_total_loss`, `forward_loss`,
`forward_joint_loss`, and `forward_tcp_loss` under `train/`, plus `critic_loss`,
`critic_total_loss`, `critic_forward_loss`, `critic_forward_joint_loss`, and
`critic_forward_tcp_loss`. Critic forward metrics are averaged across the two critics.
Auxiliary heads are omitted from actor inference exports. Older training checkpoints with
missing heads or different head inputs cannot resume this architecture; their actor exports
still work for rollout.

Continue training from a compatible configured SAC checkpoint with a fresh replay buffer:

```bash
python scripts/train_joint_policy.py experiment=continue_training
```

SAC can mix recorded real-robot transitions with simulation data. Configure recording paths
and the real-data sampling fraction under `training.real_data` in
[conf/training/default.yaml](conf/training/default.yaml). By default, training uses simulation only.

Replay the latest simulation recording, seek to a frame, and export a fixed policy-action branch request:

```bash
python scripts/replay.py
```

After exporting the request in the browser, close replay and generate the branch through the normal
episode hierarchy. Start replay again afterward to inspect the newly generated recording:

```bash
python scripts/rollout_fixed_duty.py
python scripts/replay.py
```

Convert recorded Cartesian demonstrations to a LeRobot dataset:

```bash
python scripts/convert_dataset.py +source_dir=outputs/collect_data/YYYY-MM-DD/HH-MM-SS/recordings +target_name=smolvla_waypoints
```

Fine-tune the standard pretrained SmolVLA policy with LeRobot's trainer:

```bash
python scripts/train_vla.py \
	dataset_root=datasets/smolvla_waypoints \
	steps=30000
```

Iterative teacher-labeled training uses `scripts/train_vla_dagger.py` and
`conf/train_vla_dagger.yaml`: one scripted collection round followed by VLA-only
rounds, each converted and merged before further BC updates. Checkpoints preserve
optimizer state and the initial normalization; videos remain separate when merged.

## Linux Serial Permissions

A user needs permission to open the serial device. A temporary workaround is:

```bash
sudo chmod 666 /dev/ttyACM0
```

The persistent option is to add the user to the `dialout` group, then log out and back in:

```bash
sudo usermod -aG dialout $USER
```

## One bus read per control step

On the real backend every `read_state()` is a serial round trip, and it also advances the
safety load EMA in `Arm`. The loop used to read three times per step, which both
ate bus bandwidth and made `load_ema_alpha: 0.1` behave like 0.27. It now reads once and
threads the result through:

- `Arm.get_tcp_pose(state)` takes an already-read state rather than reading for itself.
  `RealArm` runs forward kinematics on `state["Present_Position"]`; `SimArm` ignores the
  argument and reads MuJoCo directly.
- `RobotEnv.step(action, joint_positions, ...)` takes the joint positions the caller already
  observed, passed separately by `EpisodeRunner` from the current environment state.

The trade is deliberate: the commanded target is built from a reading one control period old
instead of a fresh one. At 200 Hz that is 5 ms of staleness against 1 to 3 ms of round trip
latency plus a third of the bus. Do not add a bare `read_state()` back into anything that
runs per control step.

## Local cameras

| Device | Source | Usable mode |
| --- | --- | --- |
| `/dev/video0` | ACER laptop camera | 640x480 at 30 Hz |
| `/dev/video2` | Pixel Android webcam | 640x480 at 30 Hz |

The corresponding metadata nodes, `/dev/video1` and `/dev/video3`, are not video sources.

## MuJoCo viewer

```bash
python -m mujoco.viewer --mjcf models/so101/scene.xml
```
