import numpy as np
from robot_arm.pose import Pose

p1 = Pose.from_euler([0,0,0], [0, 0, 0], 0, "XYZ", False)
p2 = Pose.from_euler([0,0,0], [np.pi/4, 0, 0], 0, "XYZ", False)

print("p1 10D:", p1.as_10d())
print("p2 10D:", p2.as_10d())
print("p1 quat (mujoco):", p1.as_mujoco_quat())
print("p2 quat (mujoco):", p2.as_mujoco_quat())

