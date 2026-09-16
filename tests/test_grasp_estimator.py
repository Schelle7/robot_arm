from types import SimpleNamespace

import pytest

from robot_arm.envs.grasp_estimator import GraspEstimator


@pytest.fixture
def estimator():
    return GraspEstimator(SimpleNamespace(
        position_range_radians=(0.3, 0.7),
        min_closing_duty=0.1,
        max_velocity_radians_per_second=0.05,
        hold_seconds=0.2,
    ))


def test_requires_continuous_stationary_closing_contact(estimator):
    assert not estimator.update(0.4, 0.0, -0.3, 0)
    assert not estimator.update(0.4, 0.0, -0.3, 199_000_000)
    assert estimator.update(0.4, 0.0, -0.3, 200_000_000)


@pytest.mark.parametrize("position,velocity,duty", [
    (0.0, 0.0, -0.3),
    (0.299, 0.0, -0.3),
    (0.8, 0.0, -0.3),
    (0.4, 0.2, -0.3),
    (0.4, 0.0, 0.3),
    (0.4, 0.0, 0.0),
])
def test_invalid_contact_clears_flag_and_restarts_timer(estimator, position, velocity, duty):
    estimator.update(0.4, 0.0, -0.3, 0)
    assert estimator.update(0.4, 0.0, -0.3, 200_000_000)
    assert not estimator.update(position, velocity, duty, 250_000_000)
    assert not estimator.update(0.4, 0.0, -0.3, 300_000_000)
    assert estimator.update(0.4, 0.0, -0.3, 500_000_000)


def test_reset_discards_previous_episode_contact(estimator):
    estimator.update(0.4, 0.0, -0.3, 0)
    assert estimator.update(0.4, 0.0, -0.3, 200_000_000)
    estimator.reset()
    assert not estimator.update(0.4, 0.0, -0.3, 1_000_000_000)


def test_lower_position_boundary_can_confirm_a_grasp(estimator):
    assert not estimator.update(0.3, 0.0, -0.3, 0)
    assert estimator.update(0.3, 0.0, -0.3, 200_000_000)
