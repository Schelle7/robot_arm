import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deployment.runpod.common import load_config, parse_run_id


class Volume:
    def __init__(self, cfg, bucket):
        self.bucket = bucket
        self.command = [
            "aws",
            "--profile",
            cfg["aws_profile"],
            "--region",
            cfg["region"],
            "--endpoint-url",
            cfg["endpoint_url"],
            "--cli-connect-timeout",
            str(cfg["connect_timeout_seconds"]),
            "--cli-read-timeout",
            str(cfg["read_timeout_seconds"]),
            "--no-cli-pager",
        ]

    def keys(self, prefix):
        result = subprocess.check_output(
            [
                *self.command,
                "s3api",
                "list-objects",
                "--bucket",
                self.bucket,
                "--prefix",
                prefix,
                "--query",
                "Contents[].Key",
                "--output",
                "json",
            ],
            text=True,
        )
        keys = json.loads(result)
        return [] if keys is None else [key for key in keys if not key.endswith("/")]

    def download(self, key, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                *self.command,
                "s3",
                "cp",
                f"s3://{self.bucket}/{key}",
                str(destination),
            ],
            check=True,
        )


def download_latest(cfg, repository):
    runs = sorted(
        (repository / cfg["local_run_dir"]).iterdir(),
        key=lambda path: parse_run_id(path.name),
        reverse=True,
    )
    run = next(path for path in runs if json.loads((path / "config.json").read_text())["training"]["type"] == cfg["training_type"])
    saved = json.loads((run / "config.json").read_text())
    run_id = saved["run_id"]
    remote = PurePosixPath(saved["training"]["persistent_root"]) / run_id
    mount = PurePosixPath(saved["pod"]["volumeMountPath"])
    prefix = remote.relative_to(mount)
    volume = Volume(cfg, saved["pod"]["networkVolumeId"])

    volume.download(str(prefix / "console.log"), run / "console.log")
    print(f"Downloaded log: {run / 'console.log'}", flush=True)
    exit_key = str(prefix / "exit_code")
    if exit_key not in volume.keys(exit_key):
        print("Final checkpoint is not available: no exit code was recorded; the run may still be training or may have stopped unexpectedly.")
        return
    volume.download(exit_key, run / "exit_code")
    exit_code = int((run / "exit_code").read_text())
    if exit_code != 0:
        print(f"Final checkpoint is not available: training failed with exit code {exit_code}. See {run / 'console.log'}.")
        return

    result_key = str(prefix / "result.json")
    if result_key not in volume.keys(result_key):
        print("Final checkpoint is not available: no training result was recorded.")
        return
    volume.download(result_key, run / "result.json")
    result = json.loads((run / "result.json").read_text())
    if result["training_type"] != cfg["training_type"]:
        raise ValueError("Training result type does not match the requested policy")
    if result["training_type"] == "joint":
        download_joint(cfg, repository, run, saved, result, volume)
        return
    if result["training_type"] != "vla_dagger":
        raise ValueError(f"Unknown training type: {result['training_type']}")
    volume.download(str(prefix / "dagger/rounds.json"), run / "rounds.json")
    checkpoint = PurePosixPath(result["checkpoint"])
    model_prefix = checkpoint.relative_to(mount) / "pretrained_model"
    keys = volume.keys(f"{model_prefix}/")
    names = {PurePosixPath(key).relative_to(model_prefix).as_posix() for key in keys}
    required = {"model.safetensors", "config.json", "policy_preprocessor.json", "policy_postprocessor.json"}
    if not required.issubset(names):
        print(f"Final checkpoint is not available: missing files {sorted(required - names)}.")
        return

    policy_root = repository / cfg["local_policy_dir"]
    destination = policy_root / f"runpod_{run_id}"
    if destination.exists():
        print(f"Checkpoint already downloaded: {destination}")
        return
    policy_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=policy_root) as temporary:
        staging = Path(temporary)
        model_dir = staging / "checkpoints/last/pretrained_model"
        for key in keys:
            relative = PurePosixPath(key).relative_to(model_prefix)
            if ".." in relative.parts:
                raise ValueError(f"Invalid checkpoint key: {key}")
            volume.download(key, model_dir / relative)
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            processor = json.loads((model_dir / name).read_text())
            for step in processor["steps"]:
                if "state_file" in step:
                    state = model_dir / step["state_file"]
                    if not state.is_file() or state.stat().st_size == 0:
                        print(f"Final checkpoint is not available: missing or empty processor state {step['state_file']}.")
                        return
        if (model_dir / "model.safetensors").stat().st_size == 0:
            raise ValueError("Downloaded model weights are empty")
        volume.download(str(prefix / "config.json"), staging / "deployment_config.json")
        volume.download(str(prefix / "dagger/.hydra/config.yaml"), staging / "dagger_config.yaml")
        round_prefix = checkpoint.parent.parent.parent.relative_to(mount)
        volume.download(str(round_prefix / "round.yaml"), staging / "round.yaml")
        for name in ("console.log", "exit_code", "rounds.json", "result.json"):
            shutil.copyfile(run / name, staging / name)
        staging.rename(destination)
    print(f"Downloaded final VLA checkpoint: {destination / 'checkpoints/last/pretrained_model'}")


def download_joint(cfg, repository, run, saved, result, volume):
    mount = PurePosixPath(saved["pod"]["volumeMountPath"])
    checkpoint = PurePosixPath(result["checkpoint"])
    source = checkpoint.parent.parent
    source_prefix = source.relative_to(mount)
    actor = checkpoint.with_suffix(".actor.npz")
    keys = [str(path.relative_to(mount)) for path in (checkpoint, actor, source / ".hydra/config.yaml")]
    available = set(volume.keys(f"{source_prefix}/"))
    missing = set(keys) - available
    model_prefix = source_prefix / "model"
    model_keys = sorted(key for key in available if key.startswith(f"{model_prefix}/"))
    if missing or not model_keys:
        print(f"Final checkpoint is not available: missing joint artifacts {sorted(missing)}; model files: {len(model_keys)}.")
        return
    timestamp = parse_run_id(saved["run_id"])
    destination = repository / cfg["local_policy_dir"] / timestamp.strftime("%Y-%m-%d/%H-%M-%S")
    if destination.exists():
        print(f"Checkpoint already downloaded: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        for key in [*keys, *model_keys]:
            relative = PurePosixPath(key).relative_to(source_prefix)
            if ".." in relative.parts:
                raise ValueError(f"Invalid checkpoint key: {key}")
            volume.download(key, staging / relative)
        for path in (checkpoint, actor):
            if (staging / "checkpoints" / path.name).stat().st_size == 0:
                raise ValueError(f"Downloaded checkpoint is empty: {path.name}")
        shutil.copyfile(run / "config.json", staging / "deployment_config.json")
        for name in ("console.log", "exit_code", "result.json"):
            shutil.copyfile(run / name, staging / name)
        staging.rename(destination)
    print(f"Downloaded final joint checkpoint: {destination / 'checkpoints' / actor.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config_path = Path(__file__).with_suffix(".toml") if args.config is None else args.config
    cfg = load_config(config_path, "download")
    download_latest(cfg, Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    main()
