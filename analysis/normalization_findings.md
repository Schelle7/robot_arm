# Failed real rollouts, 2026-09-06

## Confirmed root cause and implemented fix

The stationary hardware comparison in `outputs/hardware_modes.json` confirms that
PWM mode reports encoder ticks WITHOUT the homing offset. Position mode applies
that offset in firmware. Our RealArm adapter incorrectly treated both modes alike.
This was missing calibration in the PWM adapter, not double normalization or a
bad calibration file.

For the five body motors, `(pwm_tick - homing_offset) % 4096` matches the measured
position-mode tick EXACTLY. The gripper differs by one tick. Each original PWM
mode was restored, and torque read zero before and after the comparison.

| Joint | Old PWM interpretation | Corrected interpretation | Measured position-mode interpretation |
|---|---:|---:|---:|
| shoulder_pan | 132.220° | -49.758° | -49.758° |
| shoulder_lift | -30.374° | 64.571° | 64.571° |
| elbow_flex | -160.703° | 33.143° | 33.143° |
| wrist_flex | 35.341° | -96.879° | -96.879° |
| wrist_roll | -176.220° | 60.879° | 60.879° |
| gripper | 100.000° | 10.023° | 9.940° |

`RealArm.read_state` now restores the calibrated tick frame in PWM mode BEFORE
calling LeRobot normalization, including before the gripper is clipped. Position
mode retains its previous conversion. Modes are read at construction and the cache
is updated when `set_pwm_mode` changes them. External mode changes during a rollout
are not supported by that cache.

The raw seam in the failed run, pan 4080 -> 79, becomes calibrated ticks
2010 -> 2105: +95 ticks instead of a nearly full-turn negative jump. This fixes
that source of fictitious velocities. A calibrated full-turn wrist seam can still
exist outside the model's allowed range; this is not general multi-turn unwrapping.

Validation: 26 focused tests pass (`test_real_arm_units`, `test_analysis_hardware`,
`test_servo`), including measured mode equivalence, no double offset in position
mode, encoder seam continuity, and cache updates on PWM switching. Ruff on the
changed analysis/tests and `git diff --check` pass. The patched adapter has not
been exercised in a moving hardware rollout.

Two concerns remain distinct: PWM duty-direction evidence still needs a physical
check, and the measured calibration/model range differences remain. For example,
the current corrected wrist-flex angle (-96.879°) is 1.879° beyond the model's
-95° lower bound, but within its measured calibration. No duty polarity or model
limit change was made in this fix.

The following sections retain the earlier investigation and its intermediate
conclusions. The confirmed mode comparison above resolves their position-frame
hypothesis. Full numeric output is in `outputs/normalization_audit_confirmed.txt`.

## What was verified offline

- Read both failed episodes, including all 28 dense transitions and next observations.
- Replayed the saved NumPy actor on all 28 observations: every action matches exactly.
- Compared training/rollout frequencies, joint and TCP velocity scales, action scales,
  history durations and servo parameters: all match.
- The active LeRobot cache file `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json`
  matches `conf/hardware_calibration.json`. The repository copy is not loaded by the
  runtime factory. Neither run snapshots the calibration; historical identity is assumed.
- Installed LeRobot uses DEGREES for body joints and RANGE_0_100 for the gripper.
  The block reader extracts register words, RealArm calls `_normalize` once, and
  degrees are converted to radians once. SAC does not normalize positions again.
- Body conversion is `(tick - (range_min + range_max)/2) * 2*pi/4095`.
  LeRobot applies the homing offset in servo EEPROM, not in `_normalize`.
  Adding that offset again in Python would be incorrect in normal position mode.

## Quantified calibration/model disagreement

| Joint | Measured calibration, degrees | Model, degrees | Start at 11:24:54 | Start at 11:24:33 |
|---|---:|---:|---:|---:|
| shoulder_pan | ±119.121 | ±110.000 | 144.615 | 176.527 |
| shoulder_lift | ±103.165 | ±100.000 | 8.308 | -29.143 |
| elbow_flex | ±95.736 | ±96.830 | 132.220 | -163.780 |
| wrist_flex | ±102.769 | ±95.000 | 51.165 | 38.505 |
| wrist_roll | ±180.000 | ±157.211 | -163.912 | -144.747 |

LeRobot sets wrist_roll to 0..4095 during calibration rather than measuring its range.
Gripper 0..100 is explicitly mapped onto the model's -10..100 degrees.

The body endpoint differences are 9.12°, 3.17°, 1.09°, 7.77°, and 22.79°.
They cannot explain an elbow starting 35.39° beyond the model limit in one run and
66.95° beyond it in the other. Using 4095 instead of 4096 changes scale by only 0.0244%.

Inverting the recorded conversion gives integer ticks to within 0.00025 tick:
latest start pan=3723 (calibration maximum 3433), elbow=3412 (maximum 2997).
These are already beyond the measured calibration endpoints before the model mapping.
All reconstructed body words in both runs are 0..4095, so missing bit-15 sign decoding
does not explain their starting offsets or their observed encoder wraps.

## PWM direction evidence

In 11:24:54, substantial motion has the opposite sign to inferred guarded duty in:
pan 10/12, lift 5/7, elbow 10/11, wrist flex 11/15, wrist roll 5/6 transitions.
The earlier failed run also shows this pattern. These counts exclude |duty| < 0.1
and movements below 1°. Inertia and gravity confound individual transitions.

The clearest examples precede the large velocity artifacts:

- Elbow transitions 1 and 2: duties -0.9997 and -0.9740; angles increase 4.22° and 8.88°.
- Wrist flex transitions 1 and 2: duties +0.7986 and +0.2992; angles decrease 3.78° and 5.19°.
- Pan transitions 3 through 7: negative duties, increasingly positive angle changes.
- Wrist roll first substantial response: +0.5029 duty gives -2.20°.

The safety guard assumes positive duty increases position. With reversed PWM
direction it permits commands that drive farther out of range and blocks corrective
ones. Matching Present_Load to Goal_Time proves magnitude/encoding consistency,
not physical direction. The load echo alone does not validate this assumption.

Do not change calibration or reverse all motor commands solely from these counts:
the PWM/position direction relationship and absolute position frame need a hardware
measurement. No motion commands were issued during this investigation.

## Encoder wraps and corrections to the earlier analysis

Pan transition 8 goes from tick 4080 to 79. That is +95 encoder ticks, but the
reported angle difference is -351.736°. The environment subtracts wrapped positions
directly, generating fictitious velocities. Peak magnitude across dense inputs is
62.16 rad/s in the latest run and 109.16 rad/s in the other.

This wrap occurs at the encoder seam, not exactly at ±pi in calibrated coordinates.
The printed `obs.joint_velocities` are normalized: multiply by 1.5708 for rad/s.
Likewise `obs.tcp_velocity` is scaled; -18.5 in a rotation column is -3.7 rad/s,
not -18.5 rad/s. FK largely preserves orientation across full turns; TCP velocity
does not inherit the full joint-angle discontinuity.

Compensation includes velocity-dependent bias forces, not just gravity; the fictitious
joint velocities explain why it can saturate. Dense `terminated=True` at every fourth
step is the configured training transition boundary, not a hardware emergency stop.
The four saved Cartesian states omit the final next state after interruption; all 16
dense transitions must be read to see the gripper jump from -10° to +100° at the end.

Voltage values in old recordings are raw tenths of a volt despite the report label.
The latest run records as low as 43 (4.3 V), but the actual undervoltage threshold
was not recorded. The previous claim that it necessarily crossed a 5 V firmware
threshold is not established. Voltage units are confirmed in the vendor's
[ST3215 documentation](https://www.waveshare.com/wiki/ST3215_Servo).

## Changes and verification

- Fixed missing signed-position decoding in `read_block`, using the installed
  LeRobot decoder just like `sync_read(normalize=False)`.
- Fixed RealArm's voltage conversion to volts. Old NPZ files are unchanged.
- Added offline adapter regression tests using a real, disconnected LeRobot bus.
- Added `analysis.normalization`, which prints every transition, calibration ranges,
  reconstructed encoder counts, direction evidence and actor replay errors.
- Added `analysis.hardware`, a diagnostic that only reads registers. It bypasses
  follower configuration and does not change torque, calibration or operating mode.

The two unit fixes are valid but do not resolve the main direction/frame mismatch.
No speculative PWM sign or calibration adjustment has been applied.

Validation: `tests/test_real_arm_units.py` and `tests/test_servo.py`: 14 passed.
The broader run including `test_episode_runner.py` and `test_env_rewards.py` had
22 passes and 9 failures in existing stubs: outdated `step`/`compute_reward`
signatures and missing reset attributes. Those paths do not use the two changed
unit conversions. Ruff and `git diff --check` passed.

Reproduce the complete offline audit in the `lerobot` environment:

```bash
python -m analysis.normalization 0 1 --calibration conf/hardware_calibration.json --out outputs/normalization_audit.txt
```

Read the current hardware configuration without starting a rollout:

```bash
python -m analysis.hardware --port /dev/ttyACM0 --calibration conf/hardware_calibration.json --out outputs/hardware_units.json
```

## Successful hardware capture

The user ran the diagnostic successfully after reconnecting the external supply
to the wall. The preceding all-register timeouts were caused by missing external
power; they were not evidence of a grouped-read incompatibility.

`outputs/hardware_units.json` now confirms:

- All six stored homing offsets and min/max limits match the calibration file.
- All six motors are in PWM mode (2), with torque disabled (0), and Phase=12.
- Individual and block position reads agree exactly except for one wrist-flex tick.
- LeRobot itself normalizes the current raw readings to pan=132.220°,
  elbow=-160.791°, wrist roll=-176.220°, and gripper=100%.
- Raw elbow position is 79, outside its calibrated 819..2997 interval. Raw gripper
  position is 3907, outside 1401..2725, so LeRobot clips its output to 100%.
- The present undervoltage threshold is 40 (4.0 V), not the previously assumed 5 V.
  Present supply reads 5.2–5.3 V. This is a current snapshot, not historical proof
  of the threshold during the failed runs.

Thus the custom reader and Python unit conversion are not creating the current
position discrepancy. Stored calibration matching does not establish that PWM
feedback uses the same coordinate frame as position-mode feedback. A specific
remaining hypothesis is that PWM reports encoder position without the homing
offset: for pan, `(3582 - (-2026)) % 4096 = 1512`. A mode comparison must test this
at the same physical pose before any correction is applied.

An explicit opt-in diagnostic is now available:

```bash
python -m analysis.hardware --port /dev/ttyACM0 --calibration conf/hardware_calibration.json --compare-modes --out outputs/hardware_modes.json
```

Unlike the default read-only invocation, this writes Operating_Mode: each motor
temporarily switches between position and PWM, then returns to its original mode.
It refuses to start unless all torque-enable registers are zero. It does not write
torque enable, goals, or calibration. Restoration is attempted in a finally block
and checked by reading the mode back. Hardware communication loss can prevent
restoration; such a failure is reported explicitly. This comparison has now
been run on the physical arm; see the confirmed result at the top of this report.
Eight diagnostic tests pass, including restoration
on a failed read and refusal when torque is enabled.
