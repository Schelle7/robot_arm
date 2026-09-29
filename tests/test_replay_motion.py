import numpy as np
from omegaconf import OmegaConf

from robot_arm.replay.motion import build_motion_comparison
from robot_arm.robot_schema import POLICY_OBSERVATION_NAMES, STATE_TCP_VELOCITY_SLICE


def test_motion_comparison_uses_next_state_and_recorded_actions():
    cfg = OmegaConf.create({"waypoint": {"position_speed_meters_per_second": 0.05}, "control": {"frequencies": {"joint": 20}}})
    samples = []
    for index in range(3):
        state = np.zeros(30, dtype=np.float32)
        state[STATE_TCP_VELOCITY_SLICE] = [index + 1, -2, 3, 9, 9, 9]
        samples.append(
            {
                "obs": {name: np.array([index], dtype=np.float32) for name in POLICY_OBSERVATION_NAMES},
                "next_obs": {"state": state},
                "action": np.full(6, index, dtype=np.float32),
            }
        )

    class Policy:
        def predict_forward_motion(self, observations, actions):
            np.testing.assert_array_equal(actions[:, 0], [0, 1, 2])
            np.testing.assert_array_equal(observations["state"][:, 0], [0, 1, 2])
            predictions = np.zeros((3, 12), dtype=np.float32)
            predictions[:, 6:9] = [2, -1, 4]
            return predictions

    result = build_motion_comparison(Policy(), [samples[:2], samples[2:]], cfg)
    np.testing.assert_allclose(result["times"], [0, 0.05, 0.1])
    np.testing.assert_allclose(result["predicted"], [[0.1, -0.05, 0.2]] * 3)
    np.testing.assert_allclose(result["actual"], [[0.05, -0.1, 0.15], [0.1, -0.1, 0.15], [0.15, -0.1, 0.15]])
    np.testing.assert_allclose(result["error"], np.array(result["predicted"]) - result["actual"], atol=1e-8)
