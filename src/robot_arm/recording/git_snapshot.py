import subprocess
from pathlib import Path


def _git(*arguments: str) -> str:
    repository = Path(__file__).resolve().parents[3]
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def snapshot_git_state(output_directory: str) -> None:
    """
    Records the commit plus the full patch of everything uncommitted, so a run can be reproduced
    exactly rather than only being labelled dirty. Untracked files appear in the status by name,
    but their contents are not in the patch.
    """
    destination_directory = Path(output_directory) / "git"
    destination_directory.mkdir(parents=True, exist_ok=True)

    commit = _git("rev-parse", "HEAD").strip()
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").strip()
    status = _git("status", "--porcelain")

    (destination_directory / "status.txt").write_text(f"commit {commit}\nbranch {branch}\n\n{status}")
    (destination_directory / "uncommitted.patch").write_text(_git("diff", "HEAD"))
