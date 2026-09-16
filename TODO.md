# improve sim to real
record more data and train new policy

I have to address the frequent packet failures on the real robot arm.
my idea right now is to just do a viewer that shows response or no response as a graph
just as the current joint actions are displayed.

inverse dynamics head?
maybe add a sys id head. better randomization also according to it?
potentially certain movements for sysid



# VLA gripper control
Decide how the VLA should choose desired gripper duty and whether duty control is
enabled, instead of taking both from scripted primitives.

# eventually proper rl for vla
