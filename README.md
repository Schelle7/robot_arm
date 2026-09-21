# SO-101 follower arm

Pick-and-place with learned duty control and vision-language-action policies for
an SO-101 follower arm, using [LeRobot](https://huggingface.co/docs/lerobot) and MuJoCo.

## Simulation example

[Watch on YouTube ↗](https://youtu.be/o0uG7idjmQ8)

[![Watch: joint duty control gripping and relocating a box](https://img.youtube.com/vi/o0uG7idjmQ8/hqdefault.jpg)](https://youtu.be/o0uG7idjmQ8)

The demo uses scripted Cartesian targets and a learned joint duty controller
to pick up and relocate a box. Pick-and-place works in simulation but remains
imperfect. The demo does not demonstrate VLA-driven control.

## How control works

1. A scripted primitive generator divides a task into steps such as approach,
   close the gripper, lift, transport, and release.
2. Either a scripted Cartesian policy or a fine-tuned SmolVLA chooses Cartesian
   motion and when to advance to the next primitive. SmolVLA receives camera
   images, robot state, and the current primitive's text instruction. Its ten
   outputs are three translation channels, three rotation channels, one gripper
   position-delta channel, a primitive-completion score, desired gripper duty,
   and a duty-enable score.
3. A joint policy trained with SAC tracks the Cartesian command using robot
   state and recent history. It produces actions for the six motors, which are
   converted to applied duties by the control pipeline.
4. The arm implementation applies those duties to simulated motor dynamics in
   MuJoCo or sends PWM commands to the physical arm.

The joint controller is trained for duty control, not joint-position commands
or one-step inverse kinematics. The rollout defaults request Cartesian updates at
5 Hz, motor-control updates at 20 Hz, and simulation steps at 200 Hz.

Physical-arm sensor reading and direct PWM output are implemented. Reliable
sim-to-real transfer and real-arm communication remain ongoing work; see
[LIMITATIONS.md](LIMITATIONS.md).

## Setup

Use a conda environment with Python 3.12.13 and `ffmpeg`, and activate it before
running commands. The pinned dependencies include CUDA-enabled JAX and PyTorch.

From the repository root:

```bash
make install
```

Configuration lives in [conf/](conf/). Rollouts combine the joint training
run's saved configuration with [conf/rollout.yaml](conf/rollout.yaml).

## Run and train

| Command | Purpose |
| --- | --- |
| `make train_joint_policy` | Train the joint duty controller in simulation. |
| `make train_runpod` | Launch VLA training on RunPod using `deployment/runpod/train.toml`. |
| `make sanity_check_sim` | Roll out scripted Cartesian targets through the learned duty controller. |
| `make rollout_sim` | Roll out the VLA through the learned duty controller. |
| `make download_trained_vla` | Download the latest locally tracked RunPod run's log and, when available, final VLA checkpoint. |
| `make download_pretrained` | Download the pretrained VLA and joint controller checkpoints. |
| `make test` | Run automated tests, excluding physical sensor tests. |
| `make lint` | Check formatting and lint. |
| `make format` | Apply formatting and automatic lint fixes. |

RunPod training requires your credentials, SSH keys, network volume, and prepared
environment archive configured in [deployment/runpod/train.toml](deployment/runpod/train.toml).
The checked-in volume and archive settings refer to the author's setup.

Simulation rollouts require a trained joint checkpoint, its saved Hydra
configuration, and its model files. VLA rollouts additionally require a complete
VLA inference checkpoint. Weights are not included in the repository.

The VLA loader selects the newest complete checkpoint under
`outputs/train_vla/` or `outputs/train_vla_dagger/`. If a newer run is incomplete,
it prints that it is using an older checkpoint.

For a downloaded RunPod model, preserve this layout:

```text
outputs/train_vla/runpod_YYYY-MM-DD_HH-MM-SS/
└── checkpoints/last/pretrained_model/
    ├── model.safetensors
    ├── config.json
    ├── policy_preprocessor.json
    ├── policy_postprocessor.json
    └── ...processor state files referenced by the JSON configs
```

See [development notes](DEVELOPMENT_NOTES.md) for VLA training, dataset conversion,
recording replay, and technical details.

## Physical arm

Connect USB-C for serial communication and the DC supply for servo power.
Configure and calibrate the motors before commanding movement.

Find the serial port:

```bash
make find-port
```

Set motor IDs and baudrate only if they are not already configured, then calibrate:

```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower
```

Compare block and individual sensor reads on the connected arm:

```bash
make test-hardware PORT=/dev/ttyACM0 ID=my_follower
```

Serial permissions and hardware-specific implementation notes are in
[DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md).

## Project layout

| Directory | Contents |
| --- | --- |
| `conf/` | Hydra configuration |
| `models/` | MuJoCo robot and scene models |
| `scripts/` | Training, rollout, conversion, and replay entry points |
| `src/robot_arm/` | Control, policies, environments, recording, and replay |
| `deployment/runpod/` | Remote training and artifact downloads |
| `tests/` | Automated and hardware tests |

## Further reading

- [Development notes](DEVELOPMENT_NOTES.md): implementation decisions and detailed workflows.
- [Limitations](LIMITATIONS.md): known limitations.
- [TODO](TODO.md): planned work.
