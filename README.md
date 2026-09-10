# SO-101 follower arm

Personal control code and experiments for an SO-101 **follower** arm using
[LeRobot](https://huggingface.co/docs/lerobot).

## Simulation example

[Watch on YouTube ↗](https://youtu.be/o0uG7idjmQ8)

[![Watch: low-level duty control gripping and relocating a box](https://img.youtube.com/vi/o0uG7idjmQ8/hqdefault.jpg)](https://youtu.be/o0uG7idjmQ8)

Scripted Cartesian targets executed by the learned low-level duty controller.

## Environment

A conda env named `lerobot` (Python 3.12, with `ffmpeg`) has already been created.
Activate it before doing anything:

```bash
conda activate lerobot
```

## Install

```bash
make install
```

## Hardware checklist (before plugging in)

- USB-C cable -> controller board for logic/serial communication.
- DC power supply -> controller board. This is required to power the servos.
- On power-up the servos have torque disabled. The arm is limp until commands are sent.

## Useful Commands

Run commands with the `lerobot` environment active.

Find the USB port:

```bash
make find-port
```

Set motor IDs and baudrate, only if the motors are not already configured:

```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
```

Calibrate the follower arm:

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower
```

Check that the arm can be read:

```bash
make test PORT=/dev/ttyACM0
```

Train the low-level policy:

```bash
python scripts/train_low_level.py
```

The default experiment first treats low-level control as one-step inverse
kinematics: each terminal transition maps the current joint state and one desired
Cartesian pose delta to one joint delta. This isolates the low-level mapping
before introducing terminal multi-step action paths.

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
python scripts/train_low_level.py experiment=continue_training
```

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
	--dataset-root datasets/smolvla_waypoints \
	--output-dir outputs/train_vla/smolvla_waypoints \
	--steps 30000 \
	--batch-size 8
```

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
safety load EMA in `SafeArmWrapper`. The loop used to read three times per step, which both
ate bus bandwidth and made `load_ema_alpha: 0.1` behave like 0.27. It now reads once and
threads the result through:

- `Arm.get_tcp_pose(state)` takes an already-read state rather than reading for itself.
  `RealArm` runs forward kinematics on `state["Present_Position"]`; `SimBackend` ignores the
  argument and reads MuJoCo directly.
- `RobotEnv.step(action, joint_positions, ...)` takes the joint positions the caller already
  observed, passed separately by `EpisodeRunner` from the current environment state.

The trade is deliberate: the commanded target is built from a reading one control period old
instead of a fresh one. At 200 Hz that is 5 ms of staleness against 1 to 3 ms of round trip
latency plus a third of the bus. Do not add a bare `read_state()` back into anything that
runs per control step.

## Project Layout

```text
robot_arm/
├── conf/                    # Hydra configuration
├── models/                  # MuJoCo robot models
├── scripts/                 # Training, rollout, and data scripts
├── src/robot_arm/           # Main package
├── tests/                   # Automated tests
├── Makefile                 # Common commands
└── DEVELOPMENT_NOTES.md     # Project background and technical notes
```

For project background and accumulated technical notes, read [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md).




# Current status
I can setup the physical robot and can move its joints to the center.
Data recording position, temperature and pwm readings work.

The simulation can learn a inverse kinematics control from 3D position, 3D rotation, and 1D gripper state to joint control.
I will later extend this to use pwm control on the real robot and hope to achieve smooth movement.


I will focus next on smol vla and etaching it some tasks.
smolvla is there to define the desired path in 3d space + 3d orientation + 1d gripepr open/closed

I also want to work on function calling and teaching llms that so I will probably add a third level a general llm that can tell the vla what to do. Like grab box. move it to the right and so on. (as function calls).
