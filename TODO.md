# improve sim to real
inverse dynamics head?
maybe add a sys id head. better randomization also according to it?
potentially certain movements for sysid

I have to address the frequent packet failures on the real robot arm.


# VLA
a bit sensible behavior but I need to fix the action  completion
probably increase completion ratio probability with inverse of distance for waypoints
and something fancier for the box gripping
or maybe distance + gripping requirement

probably mix vla executions with recovering from stupid behavior
then more BC
implement this loop tomorrow
I need a new label
cartesian expert action and actually executed action
decide what to do about success filter, teh current version ain't gonna work.
what do I do about normalization?
what amount of precision do I need for the transition to the next primitive?
