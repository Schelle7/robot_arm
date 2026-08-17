# 5Hz VLA


# Low level
(not strictly low level)
When VLA-generated paths are introduced, they may self-intersect or pass close to earlier segments, so path following must stop relying on global nearest-segment selection and track progress sequentially.

think about how what kind of path tracking can be sensibly accomplished.

Recheck whether waypoint rotation-vector reachability thresholds and reward axis-distance metrics should remain different.
Yes. A debug replay should either display recorded facts or call the same domain logic used during rollout. This replay currently mixes recorded facts, shared visualization helpers, and independent reconstructions without clearly distinguishing them.



consider sampling joint positions, then doing forward kinematics, filtering invalid poses (self collission and maybe collision with ground / maybe deactivate ground)



# get inverse dynamics / kinematics working on the real robot.

# look at the conversion from my data to vla lerobot format again and then actually train it
squash git commits, make the repo public and so on


the previously discovered random waypoint pose ranges got lost mostly
Maybe I should remember them


Maybe the privileged end effector pos should be renamed? since fk kinematics is reasonably accurate on real robot.





You are identifying a real missing signal, but I would not add force to Pose.

A pose describes geometry:

Position
Orientation
Gripper opening
Force is measured interaction state, not pose. The same pose may have:

Zero force when the gripper is empty.
Positive force while holding an object.
Excessive force when stalled.
Putting force in Pose would make methods such as delta_to(), apply_delta(), waypoint interpolation, and pose-completion checks ambiguous. What is a “force delta” between two poses? It does not belong in the same mathematical object.

the gripper closed part needs some form of force and then it cannot and does not even want to close component.

I have to think about how to implement this.
