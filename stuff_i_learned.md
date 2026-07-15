PWM stands for pulse width modulation.
digital microcontrollers can only output 1 or 0.
Therefore they have to switch between 0 and 1 many times to simulate 50% or any farction of motor energy / velocity.
for 50% its on off on off
and so on for 70% its on 70% of the time and off 30%


Apparently the servos from so 101 can receive a new percentage signal at 500-1000Hz

gemini recommends a control frequency of 100Hz


you can read on register 60-61? what current is applied apaprently
but only for all of the engines combined I believe so kinda useless

you can also read a velocity estimate

reading takes some of the bandwidth otherwise used for writing the pwm amount.


There might be multiple modes about what to do based on the state estimation guess.



Alltogether a bit unclear whether it is useful to use the pwm. Only really makes sense if I have an IMU or soemthing I guess.

Maybe I'll just try the VLA for now.
velocity would be helpful though.

Have to consider.



USB-serial latency timer. FTDI/CH340/CP210x adapters buffer incoming bytes up to a timer (default often 16ms) before delivering them to userspace. That alone caps you near 60Hz round-trip and adds jitter. On Linux you drop it via the driver's latency_timer sysfs attribute. This is the number one fix.

Half-duplex round trips. Each sync_read is request + servo response. At 1Mbaud a 6-servo sync read is on the order of a few hundred microseconds of wire time, but the round-trip latency (turnaround + USB + timer) dominates. Reading every cycle costs you bandwidth you could spend writing.

Python/GIL/scheduling jitter. time.sleep is not a real-time clock. At 100Hz (10ms budget) a scheduling hiccup is a large fraction of your cycle.

Doing read + write every step. Your loop writes Goal_Position then immediately reads Present_Position each iteration. If you only need position feedback for logging, reading less often frees the bus.



use_relative_actions: bool = False
relative_exclude_joints: list[str] = ["gripper"]

Supported for the pi family (pi0, pi0.5, pi0_fast). When enabled, the pipeline does:

gives a delta apparently so somewhat close to velocity
/scaled velocity

check for existing checkpoints that have trained with relative psoition.


record pwm and tempearture and position values when trying such a trained policy.

Then incorporate that custom data into the policy and the precision increases hopefully when redeploying with the adapter.

Also record the pwm of that execution.

retrain the pwm with the new data
(Im intending some desired trajectory vs what the pwm did kind of training have to decide exactly what I am going to do, maybe some machine elarning MPC kind of thing)
, or inverse dynamics
If it does a retraining the VLA might be sensible since the behavior of the arm has changed.

feed temperature and current and maybe voltage to the neural net?

also set up some safety stuff like limiting the pwm with temperature or for other reasons like velocity.




Max_Temperature_Limit (default around 70C). Combined with an unloading-condition bitmask that selects which faults cut torque, over-temperature normally triggers a torque unload. The servo shuts its own output off and raises an error flag when it gets too hot.
Overload / over-current protection: a load/current threshold plus a protection-time register. Sustained excessive load drops it to reduced torque or cuts output.
Over-voltage / under-voltage protection similarly.
this kinda contradicts Opus previous opinion that we have to read the temperature to stop the arm since it can do that by itself.
Reading the temperature is useful for the low level controller and for gracefully slowing down still so there is some truth in that we should read it.


a stall/jam is something the servo cannot really handle itself.
it's a bit like cobots that should slow down before getting close to humans.

