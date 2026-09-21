import argparse
import contextlib
import sys
from pathlib import Path

from diagnostics.report import print_rollout
from diagnostics.rollouts import ROLLOUT_ROOT, find_rollouts, load_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="List recent rollouts, or dump one of them in full.")
    parser.add_argument("run", nargs="?", help="Run directory, or an index into the listing where 0 is the newest.")
    parser.add_argument("--limit", type=int, default=10, help="Runs to list.")
    parser.add_argument("--out", type=Path, help="Write the dump here instead of stdout. A full run is thousands of lines.")
    args = parser.parse_args()

    runs = find_rollouts(limit=args.limit, root=ROLLOUT_ROOT)

    if args.run is None:
        for index, run in enumerate(runs):
            episode_count = sum(1 for _ in run.glob("**/episode.npz"))
            print(f"{index:>3}  {run}  {episode_count} episodes")
        return

    run_dir = runs[int(args.run)] if args.run.isdigit() else Path(args.run)
    rollout = load_rollout(run_dir)

    if args.out is None:
        print_rollout(rollout)
        return

    with args.out.open("w") as handle, contextlib.redirect_stdout(handle):
        print_rollout(rollout)
    print(f"Wrote {args.out} ({sum(1 for _ in args.out.open())} lines)", file=sys.stderr)


if __name__ == "__main__":
    main()
