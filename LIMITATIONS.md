# Limitations

Desired gripper duty and its enable flag are supplied by scripted primitives,
not predicted by the VLA. The VLA predicts a gripper position delta, while the
joint controller receives the duty target and mode separately. Choosing grip
effort and when to use duty control therefore still depends on the scripted task.

The seven-dimensional Cartesian action projections are trained from scratch; their rollout performance still needs evaluation.

Random waypoint targets are sampled in Cartesian space with no reachability or collision check, so some ask for a position and orientation the 5-DOF arm cannot hold at once, and some can only be approached by moving through the floor.
(the sampling is designed to avoid it but it isnt 100% guaranteed)


My robot arm is considerably heavier than the version in sim.
704 grams including the servo controller board and cables excluding the camera mount and camera vs
according to Astra 644g with camera mount without cables or servo board or camera.
however I have some weight randomization.
Can consider in the future.


my current setup is a bit dumb.
Since I have the orange border around the tile, the color where its supposed to move probably will be ignored.
