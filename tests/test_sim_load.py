import mujoco
import numpy as np


def test_so101_sim_load():
    model_path = "models/so101/scene.xml"
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    print(f"\n--- Loading {model_path} ---")
    print(f"Total Joints: {m.njnt}")
    print(f"Total Actuators: {m.nu}")

    print("\nActuator Mapping & Limits (in radians):")
    for i in range(m.nu):
        actuator_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        jnt_id = m.actuator_trnid[i, 0]
        jnt_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)

        ctrl_range = m.actuator_ctrlrange[i]
        jnt_range = m.jnt_range[jnt_id]

        print(f"[{i}] Actuator: {actuator_name} -> Joint: {jnt_name}")
        print(f"    Control Range: {ctrl_range[0]:.4f} to {ctrl_range[1]:.4f}")
        print(f"    Joint Range:   {jnt_range[0]:.4f} to {jnt_range[1]:.4f}")

    # Check structural expectations
    assert m.nu == 6, f"Expected 6 actuators, got {m.nu}"

    # Do a basic forward step to verify physics don't instantly explode
    mujoco.mj_step(m, d)

    assert not np.any(np.isnan(d.qpos)), "Simulation exploded: NaNs in positions"
    assert not np.any(np.isnan(d.qvel)), "Simulation exploded: NaNs in velocities"

    print("\nForward physics step passed (no NaNs).")
