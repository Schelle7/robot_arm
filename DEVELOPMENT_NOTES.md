# Development notes

## Control stack

1. SmolVLA maps camera images, prompts, and proprioception to Cartesian pose deltas and primitive completion.
2. SAC maps Cartesian tracking state to joint-duty actions centered on model-derived compensation.
3. The backend applies duties in MuJoCo or communicates with the physical SO-101.

Pick-and-place uses separate open, approach, close, lift, transport, lower, and release primitives. Visual primitives retain privileged targets for supervision but do not expose target offsets to SmolVLA.

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
