# Development notes

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
- Real-arm direct PWM duty output is not implemented.

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
