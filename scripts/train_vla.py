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
            "--policy.type=cartesian_smolvla",
            "--policy.pretrained_path=lerobot/smolvla_base",
            "--policy.discover_packages_path=robot_arm.cartesian_smolvla",
            "--policy.n_action_steps=1",
            f"--dataset.repo_id={dataset_root.name}",
            f"--dataset.root={dataset_root}",
            f"--output_dir={output_dir}",
            "--job_name=robot_arm_smolvla",
            f"--steps={args.steps}",
            f"--batch_size={args.batch_size}",
            "--policy.push_to_hub=false",
            "--wandb.enable=false",
        ],
        check=True,
    )
    project_root = Path(__file__).resolve().parent.parent
    latest_run_file = project_root / "outputs" / "train_vla" / "latest_run.txt"
    latest_run_file.write_text(str(output_dir))


if __name__ == "__main__":
    main()