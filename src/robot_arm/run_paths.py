from datetime import datetime
from pathlib import Path


def run_timestamp(run_directory: Path) -> datetime:
    if run_directory.name.startswith("runpod_"):
        timestamp = run_directory.name.removeprefix("runpod_")
        timestamp_format = "%Y-%m-%d_%H-%M-%S" if "_" in timestamp else "%Y-%m-%d"
        return datetime.strptime(timestamp, timestamp_format)
    return datetime.strptime(f"{run_directory.parent.name}/{run_directory.name}", "%Y-%m-%d/%H-%M-%S")


def find_runs(roots: list[Path]) -> list[Path]:
    runs = [run for root in roots for pattern in ("????-??-??/??-??-??", "runpod_*") for run in root.glob(pattern) if run.is_dir()]
    return sorted(runs, key=lambda run: (run_timestamp(run), run), reverse=True)
