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

Installs this project plus `lerobot[feetech]` (Feetech STS3215 motor support) into
the active `lerobot` env:

```bash
make install
```

## Hardware checklist (before plugging in)

- USB-C cable: computer -> controller board (logic/serial).
- DC power supply -> controller board (**required** to power the servos).
- On power-up the servos come up with torque **disabled**: the arm is limp and
  will not move on its own. It only moves when a command is sent.

## Remaining steps

Order matters. Run each with the `lerobot` env active.

### 1. Find the USB port

```bash
make find-port          # wraps: lerobot-find-port
```

Note the reported device, e.g. `/dev/ttyACM0`.

#### Serial port permissions (Linux)

On Linux your user needs permission to open the serial device, otherwise every
connection fails with `PermissionError: [Errno 13] Permission denied` (which can
masquerade as a "cannot connect / wrong port" error).

Temporary (resets when the device is replugged or on reboot):

```bash
sudo chmod 666 /dev/ttyACM0
```

Permanent (recommended) — add your user to the `dialout` group once, then log
out and back in for it to take effect:

```bash
sudo usermod -aG dialout $USER
```

### 2. Set motor IDs and baudrate (once per set of motors)

Only needed if the motors were **not** pre-configured with IDs 1-6.
Requires connecting **one motor at a time** to the controller board, so you must
unplug the 3-pin daisy-chain cables between joints and step through them
(gripper -> wrist_roll -> wrist_flex -> elbow_flex -> shoulder_lift -> shoulder_pan),
then re-chain them afterwards. No disassembly of the arm is required.

```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
```

Skip this step if the arm already talks to all 6 motors (see step 4).

### 3. Calibrate

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower
```

### 4. Sanity check

Reads motor positions from the follower to confirm the bus is working:

```bash
make test PORT=/dev/ttyACM0
```

## Layout

```
robot_arm/
├── pyproject.toml      # this project; depends on lerobot[feetech]
├── Makefile            # install / find-port / test helpers
└── src/robot_arm/
    ├── __init__.py
    └── check_arm.py    # connects to the follower and prints motor positions
```



Node	Device	Usable
video0	ACER HD User Facing (built-in laptop webcam)	yes, 640x480@30
video1	ACER (metadata node)	no
video2	Pixel 10 Pro: Android Webcam	yes, 640x480@30
video3	Pixel (metadata node)	no



visualize the arm:
python -m mujoco.viewer --mjcf models/so101/scene.xml



### 5.
plans:
smolvla + custom 7d pose to motor commands adapter
currently the adapter outputs delta positions which then get converted to position targets.
Later on I want to output PWM values.


set up a training loop for vla training based on synthetic and real trajectories.
For training it makes sense if I use the lerobot data format
Then I don't have to handle any of it and can just call the existing training script.


### Safety layer
(long term pwm safety)
check the allowed ticks and then only allow certain goal positions.
The safety layer checks the commands and decides what to execute.
Does the command respect the defined boundaries?

What is the current velocity of the motor?
What is its current temperature?




### longer term plans
(at some point I want to set up adaptation to the specific robot via the past transitions and a latent sysid vector)
