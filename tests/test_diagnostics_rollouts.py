import numpy as np

from diagnostics.rollouts import find_rollouts, load_rollout


def make_run(root, job, date, time):
    run = root / job / date / time
    (run / ".hydra").mkdir(parents=True)
    (run / ".hydra/config.yaml").write_text("model_path: models/scene.xml\n")
    return run


def test_find_rollouts_orders_jobs_together_and_keeps_runs_without_episodes(tmp_path):
    older = make_run(tmp_path, "rollout_vla", "2026-09-18", "12-00-00")
    newer = make_run(tmp_path, "rollout_waypoint", "2026-09-19", "09-00-00")
    (tmp_path / "rollout_vla/2026-09-20/12-00-00").mkdir(parents=True)

    assert find_rollouts(limit=10, root=tmp_path) == [newer, older]
    assert find_rollouts(limit=1, root=tmp_path) == [newer]


def test_load_rollout_keeps_every_episode_separate(tmp_path):
    run = make_run(tmp_path, "rollout_vla", "2026-09-18", "12-00-00")
    paths = []
    for index in (2, 1):
        path = run / "recordings" / f"episode_{index}" / "episode.npz"
        path.parent.mkdir(parents=True)
        np.savez(path, reward=np.array([index], dtype=np.float32))
        paths.append(path)

    rollout = load_rollout(run)

    assert list(rollout.episodes) == sorted(paths)
    for index, data in enumerate(rollout.episodes.values(), start=1):
        np.testing.assert_array_equal(data["reward"], [index])


def test_load_rollout_without_recordings(tmp_path):
    run = make_run(tmp_path, "rollout_vla", "2026-09-18", "12-00-00")

    assert load_rollout(run).episodes == {}
