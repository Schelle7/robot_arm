# improve sim to real
record more data and train new policy

I have to address the frequent packet failures on the real robot arm.
my idea right now is to just do a viewer that shows response or no response as a graph
just as the current joint actions are displayed.

probably record and eventually add to joint policy whether a read has failed.


inverse dynamics head?
maybe add a sys id head. better randomization also according to it?
potentially certain movements for sysid


# VLA gripper control
Collect recordings with gripper-duty labels, convert a new ten-dimensional action dataset,
and train and evaluate the VLA's predicted duty and duty-enable outputs.


implement action chunks? of cartesian actions.

# eventually proper rl for vla


I am currently not saving all sensor readings.
