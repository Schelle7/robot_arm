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




### 5.
decide on some vla and then try to run it.
collect some data and then further train a policy.


visualize the arm:
python -m mujoco.viewer --mjcf models/so101/scene.xml



Long term I want to setup my behavior cloning / RL loop with head that can do inverse kinematics and forward dynamics (does forward dynamics a la MPC with neural net help? Does it fight sim to real gap? and help better understand the timing mismatch?)


short term just set up the normal smol VLA with PID.






next step probably, add a red box and tell the gripper to grip it.
check if it does, maybe the camera alignment is wrong or something.

set up a training loop for training based on synthetic and vla trajectories.
For training it makes sense if I use the lerobot data format
Then I don't have to handle any of it and can just call the existing training script.

Try a run on the real arm.



I could actually introduce a safety layer akin to the one for the excavator that decides what actions will be permitted.
Probably just have to make sure the position decided by the VLA is inside the allowed range.
Then also not too high a speed?
And some temperature checks??
That should make it safe to try out on the real arm.



maybe I should build an RL world after all sicne so far I dont really have any way to properly program the robot.
doing inverse kinematics by hand is a bit of a pain.

but maybe still easier?
6d pose is far easier to obtain + gripper movement.

Therefore I think I am going to try that.
and then RL and no inverse kinematics.



tool center point in between the two gripepr parts + end effector pos as the non movimng gripper part?
aperture can be abgeleitet from the servo position of the gripper and is therefore not needed separately I think.


but maybe it is for the high level policy?
or should high level policy specify that part directly?


reward
scaling_vector * (|goal_pose - previous_pose| - |goal_pose - new_pose|)


lets please assume the high level planner provides a 7 dimensional vector 6d psoe + gripper and an array for teh enxt trajectories of those.

Then we have to set up a reward structure that tells how well the trajectory ahs been acheived.
one important aspect is that you dont have to necessarily acheive the whole trajectory.
Achieving teh first 70% of it well is equally good (then teh low level policy has some flexibility)


repalce the action head of high level vla by 7d * n trajectory poses
(7d = 6d pose + gripper)
apparently smolVLA autoadapts to the size from teh dataset.


completely train that and then for the rest probably do lora only.







later on I want to add pwm including considering temperature and velocity.
for now it is just position.


Regarding the LeRobot calibration file: the calibration limits are usually saved heavily nested inside your huggingface cache or dot config (e.g., ~/.cache/huggingface/lerobot/ or ~/.lerobot/ usually under a .json configuration linking motor IDs to their exact hardware tick offsets).

For now, passing the basic hardcoded defaults in the yaml works exactly as intended until you map the real dynamic parsing!
need to setup a safety wrapper before palcing it on the real arm.
