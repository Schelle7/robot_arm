# SO-101 follower arm

Personal control code and experiments for an SO-101 **follower** arm using
[LeRobot](https://huggingface.co/docs/lerobot).

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

The `debug` experiment first treats low-level control as one-step inverse
kinematics: each terminal transition maps the current joint state and one desired
Cartesian pose delta to one joint delta. This isolates the low-level mapping
before introducing terminal multi-step action paths.

```bash
python scripts/train_low_level.py experiment=debug
```

Continue training from the configured SAC checkpoint with a fresh replay buffer:

```bash
python scripts/train_low_level.py experiment=continue_training
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

I also want to work on function calling and etaching llms that so I will probably add a third level a general llm that can tell the vla what to do. Like grab box. move it to the right and so on. (as function calls).
