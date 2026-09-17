from datetime import datetime
from pathlib import Path

import pytest

from robot_arm.run_paths import find_runs, run_timestamp


@pytest.mark.parametrize("path, expected", [
    ("outputs/train_joint_policy/2026-09-16/12-34-56", datetime(2026, 9, 16, 12, 34, 56)),
    ("outputs/train_vla/runpod_2026-09-16_12-34-56", datetime(2026, 9, 16, 12, 34, 56)),
    ("outputs/train_vla/runpod_2026-09-16", datetime(2026, 9, 16)),
])
def test_run_timestamp(path, expected):
    assert run_timestamp(Path(path)) == expected


def test_find_runs_orders_all_roots_and_layouts_by_date(tmp_path):
    roots = [tmp_path / "train_vla", tmp_path / "train_vla_dagger"]
    newest = roots[1] / "2026-09-17/10-00-00"
    middle = roots[0] / "runpod_2026-09-16_12-34-56"
    oldest = roots[0] / "2026-09-15/18-00-00"
    for run in (oldest, newest, middle):
        run.mkdir(parents=True)
    (roots[0] / "runpod_2026-09-18").write_text("not a directory")
    (roots[0] / ".download-pending").mkdir()
    assert find_runs(roots) == [newest, middle, oldest]
