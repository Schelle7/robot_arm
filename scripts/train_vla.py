import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    subprocess.run(
        [
            "lerobot-train",
            "--policy.path=lerobot/smolvla_base",
            f"--dataset.repo_id={dataset_root.name}",
            f"--dataset.root={dataset_root}",
            f"--output_dir={output_dir}",
            "--job_name=robot_arm_smolvla",
            f"--steps={args.steps}",
            f"--batch_size={args.batch_size}",
            "--wandb.enable=false",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()