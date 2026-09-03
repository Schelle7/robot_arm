# how do I best sample positions?
consider sampling joint positions, then doing forward kinematics, filtering invalid poses (self collission and maybe collision with ground / maybe deactivate ground)

# randomization delay for sim is necessary eventually

# maybe add an action smoothing penalty and the last action(s)



(lerobot) jelle@jelle:~/Desktop/robot_arm$ python scripts/rollout_waypoint.py 
Loading low level policy from: /home/jelle/Desktop/robot_arm/outputs/train_low_level/2026-09-05/11-02-06/checkpoints/jax_sac_final_240228.pkl
Saved to: /home/jelle/Desktop/robot_arm/outputs/rollout/rollout_waypoint/2026-09-05/13-23-39/waypoint_recording/waypoint_sanity_check/episode.npz

a quite interesting failure mode

the arm hits the ground and can't achieve the desired pose.
