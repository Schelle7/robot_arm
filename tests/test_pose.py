import numpy as np
from robot_arm.pose import Pose

def test_10d_roundtrip():
    # Test rotation around X
    p_x = Pose.from_euler([0,0,0], [np.pi/4, 0, 0], 0, "XYZ", False)
    p_x_rt = Pose.from_10d(p_x.as_10d())
    np.testing.assert_allclose(p_x.as_mujoco_quat(), p_x_rt.as_mujoco_quat(), atol=1e-5)
    
    # Test rotation around Y
    p_y = Pose.from_euler([0,0,0], [0, np.pi/4, 0], 0, "XYZ", False)
    p_y_rt = Pose.from_10d(p_y.as_10d())
    np.testing.assert_allclose(p_y.as_mujoco_quat(), p_y_rt.as_mujoco_quat(), atol=1e-5)
    
    # Test rotation around Z
    p_z = Pose.from_euler([0,0,0], [0, 0, np.pi/4], 0, "XYZ", False)
    p_z_rt = Pose.from_10d(p_z.as_10d())
    np.testing.assert_allclose(p_z.as_mujoco_quat(), p_z_rt.as_mujoco_quat(), atol=1e-5)

    # Test full matrix roundtrip to verify 6D continuous representation properties
    p_multi = Pose.from_euler([1, 2, 3], [np.pi/3, -np.pi/4, np.pi/6], 0.5, "XYZ", False)
    p_multi_rt = Pose.from_10d(p_multi.as_10d())
    np.testing.assert_allclose(p_multi.as_matrix(), p_multi_rt.as_matrix(), atol=1e-5)
    
    # Test converting a from_10d pose back to 10D to check state mutation
    np.testing.assert_allclose(p_multi.as_10d(), p_multi_rt.as_10d(), atol=1e-5)

def test_10d_uniqueness():
    p1 = Pose.from_euler([0,0,0], [-np.pi/4, np.pi/4, np.pi/6], 0, "XYZ", False)
    p2 = Pose.from_euler([0,0,0], [np.pi/4, np.pi/4, np.pi/6], 0, "XYZ", False)
    p3 = Pose.from_euler([0,0,0], [0, np.pi/4, np.pi/6], 0, "XYZ", False)
    
    assert not np.allclose(p1.as_10d(), p2.as_10d())
    assert not np.allclose(p1.as_10d(), p3.as_10d())
    assert not np.allclose(p2.as_10d(), p3.as_10d())
    print("All different")

if __name__ == "__main__":
    test_10d_roundtrip()
    test_10d_uniqueness()
    print("All roundtrip tests passed! Quaternions are perfectly preserved.")
