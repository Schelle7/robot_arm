consider sampling joint positions, then doing forward kinematics, filtering invalid poses (self collission and maybe collision with ground / maybe deactivate ground)


## 1. Missing Primitives

- **`close_gripper` / `open_gripper`**
  - Prompt: `"close gripper"` / `"open gripper"`
  - Target: Keep TCP position fixed, transition gripper to target closed/open limit.
  - Currently fused into `"close gripper and lift {color} box"`.

- **`pick_up(color)`**
  - Prompt: `"pick up {color} box"`
  - Target: Move TCP vertically upwards (+z) while keeping gripper closed.
  - Same fusion.


## 3. Decisions to make

- Does the privileged target offset stay in `observation.state` for the vision primitives? While it
  is there, `"move to red box"` is solvable without looking at the image at all.

I guess the clear anwser is no?
The only doubt is how the primitive policy can deal with it?


velocities in sim and real are measured differently
As long as we do not have a working sim with pwm it doesnt matter but later should be checked.
