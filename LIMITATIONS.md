# Limitations

Random waypoint targets are sampled in Cartesian space with no reachability or collision check, so some ask for a position and orientation the 5-DOF arm cannot hold at once, and some can only be approached by moving through the floor.
(the sampling is designed to avoid it but it isnt 100% guaranteed)
