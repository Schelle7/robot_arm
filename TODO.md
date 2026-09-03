# how do I best sample positions?
consider sampling joint positions, then doing forward kinematics, filtering invalid poses (self collission and maybe collision with ground / maybe deactivate ground)

# measure velocities identically in sim and real
velocities in sim and real are measured differently
As long as we do not have a working sim with pwm it doesn't matter but later should be checked.
Change once we have a working 20Hz or more simulation.
probably define the amount of desired time used for velocity calculation
