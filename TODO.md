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

implement action chunks? of cartesian actions.

# eventually proper rl for vla



I started moving the arm safety code into arm instead.
I suppose there might be some duplicated methods in the separate arms, check at some point.
