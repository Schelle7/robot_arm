import numpy as np
import pytest
from omegaconf import OmegaConf

from robot_arm.data.transition_loader import load_real_transitions
from robot_arm.robot_schema import MOTOR_ORDER
from robot_arm.training.replay_buffer import NumpyReplayBuffer


@pytest.fixture
def recording_case(tmp_path):
    cfg = OmegaConf.create(
        {
            name: OmegaConf.to_container(OmegaConf.load(f"conf/{name}/default.yaml"), resolve=True)
            for name in ("control", "waypoint", "servo", "safety", "reward", "training", "runtime")
        }
    )
    cfg.arm_type = "sim"
    recorded_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    recorded_cfg.arm_type = "real"
    recorded_cfg.runtime.record_policy_debug = True
    run = tmp_path / "session"
    (run / ".hydra").mkdir(parents=True)
    episode = run / "recordings" / "episode_0000" / "episode.npz"
    episode.parent.mkdir(parents=True)
    config_path = run / ".hydra" / "config.yaml"
    OmegaConf.save(recorded_cfg, config_path)
    sizes = {"history": 3, "state": 2, "goal": 1}
    step = {
        "obs": {name: np.zeros(size) for name, size in sizes.items()},
        "next_obs": {name: np.ones(size) for name, size in sizes.items()},
        "action": np.full(6, 0.5),
        "requested_duty": np.array([cfg.servo.max_duty[name] / cfg.servo.full_scale_duty for name in MOTOR_ORDER]) * 0.5,
        "reward": 2.0,
        "terminated": True,
    }
    np.savez_compressed(episode, dense_trajectory=np.array([[step]], dtype=object))
    return cfg, recorded_cfg, config_path, episode, step, NumpyReplayBuffer(2, sizes, 6, 42)


def test_loads_current_real_transition(recording_case):
    cfg, _, _, episode, _, buffer = recording_case
    assert load_real_transitions([str(episode)], cfg, buffer) == 1
    assert buffer.size == 1
    assert buffer.rewards[0, 0] == 2.0
    assert buffer.dones[0, 0] == 1.0
    np.testing.assert_array_equal(buffer.actions[0], np.full(6, 0.5))


@pytest.mark.parametrize("field,value", [("arm_type", "sim"), ("control.frequencies.joint", 123), ("reward.action_change_penalty_factor", 123)])
def test_rejects_incompatible_recording_configuration(recording_case, field, value):
    cfg, recorded_cfg, config_path, episode, _, buffer = recording_case
    OmegaConf.update(recorded_cfg, field, value)
    OmegaConf.save(recorded_cfg, config_path)
    with pytest.raises((ValueError, AssertionError)):
        load_real_transitions([str(episode)], cfg, buffer)
    assert buffer.size == 0


def test_requires_current_duty_field(recording_case):
    cfg, _, _, episode, step, buffer = recording_case
    del step["requested_duty"]
    np.savez_compressed(episode, dense_trajectory=np.array([[step]], dtype=object))
    with pytest.raises(KeyError, match="requested_duty"):
        load_real_transitions([str(episode)], cfg, buffer)


def test_rejects_wrong_observation_shape(recording_case):
    cfg, _, _, episode, step, buffer = recording_case
    step["obs"]["state"] = np.zeros(1)
    np.savez_compressed(episode, dense_trajectory=np.array([[step]], dtype=object))
    with pytest.raises(AssertionError, match="shape"):
        load_real_transitions([str(episode)], cfg, buffer)


def test_rejects_real_capacity_overflow(recording_case):
    cfg, _, _, episode, step, buffer = recording_case
    np.savez_compressed(episode, dense_trajectory=np.array([[step, step, step]], dtype=object))
    with pytest.raises(ValueError, match="capacity"):
        load_real_transitions([str(episode)], cfg, buffer)
    assert buffer.size == 0


def test_rejects_duty_that_disagrees_with_policy_action(recording_case):
    cfg, _, _, episode, step, buffer = recording_case
    step["requested_duty"] = np.zeros(6)
    np.savez_compressed(episode, dense_trajectory=np.array([[step]], dtype=object))
    with pytest.raises(AssertionError, match="direct-duty scaling"):
        load_real_transitions([str(episode)], cfg, buffer)


def test_rejects_non_finite_observations(recording_case):
    cfg, _, _, episode, step, buffer = recording_case
    step["next_obs"]["state"][0] = np.nan
    np.savez_compressed(episode, dense_trajectory=np.array([[step]], dtype=object))
    with pytest.raises(AssertionError, match="Non-finite"):
        load_real_transitions([str(episode)], cfg, buffer)


def test_rejects_duplicate_episode_paths(recording_case):
    cfg, _, _, episode, _, buffer = recording_case
    with pytest.raises(AssertionError, match="Duplicate"):
        load_real_transitions([str(episode), str(episode)], cfg, buffer)
