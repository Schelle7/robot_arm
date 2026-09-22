import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deployment.runpod.common import PodUnavailable, api_request, create_local_run, load_api_key, load_config, print_time, run_stage
from deployment.runpod.prepare_environment import wait_for_ssh


def bootstrap(cfg):
    root_path = Path(cfg["training"]["local_root"])
    destination_path = Path(cfg["training"]["persistent_root"]) / cfg["run_id"]
    root = shlex.quote(str(root_path))
    destination = shlex.quote(str(destination_path))
    repo = shlex.quote(str(root_path / "repo"))
    exit_code = shlex.quote(str(destination_path / "exit_code"))
    log_file = shlex.quote(str(destination_path / "console.log"))
    config_file = shlex.quote(str(destination_path / "config.json"))
    volume = shlex.quote(cfg["pod"]["volumeMountPath"])
    repository_url = shlex.quote(cfg["training"]["repository_url"])
    commit = shlex.quote(cfg["commit"])
    api_url = shlex.quote(cfg["api_url"] + "/pods/")
    timeout = cfg["request_timeout_seconds"]
    cleanup = f"""finish() {{
    code=$?
    trap - EXIT
    echo "Training process ended with exit code $code; terminating Pod"
    if [ -d {destination} ]; then
        printf '%s\\n' "$code" > {exit_code}
    fi
    curl --fail --silent --show-error --max-time {timeout} \\
        --request DELETE --header "Authorization: Bearer $RUNPOD_API_KEY" \\
        {api_url}"$RUNPOD_POD_ID"
    exit "$code"
}}
trap finish EXIT
"""

    logging = f"""mountpoint -q {volume}
mkdir -p {destination}
exec > >(tee -a {log_file}) 2>&1
printf '%s\\n' "$TRAIN_CONFIG" > {config_file}
"""

    installation = f"""echo 'Installing system libraries for headless rendering and Git'
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates libegl1 libgl1 libopengl0 openssh-server
mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh
printf '%s\\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A
/usr/sbin/sshd
mkdir -p {root}
echo 'Cloning training repository'
git clone -- {repository_url} {repo}
cd {repo}
git checkout --detach {commit}
"""
    training = f"python3 -u deployment/runpod/train.py worker --config {config_file}\n"

    return "\n".join(["set -euo pipefail", cleanup, logging, installation, training])


def restore_environment(cfg, env):
    root = Path(cfg["training"]["local_root"])
    prefix = root / "environment"
    prefix.mkdir()
    if root.stat().st_dev == Path(cfg["pod"]["volumeMountPath"]).stat().st_dev:
        raise RuntimeError("The training environment must be extracted onto local disk")
    run_stage("extract-environment", ["tar", "-xzf", cfg["training"]["archive"], "-C", str(prefix)], env)
    env["PATH"] = f"{prefix / 'bin'}:{env['PATH']}"
    run_stage("relocate-environment", [str(prefix / "bin/python"), str(prefix / "bin/conda-unpack")], env)
    return prefix


def training_command(cfg, python, repo, destination):
    kind = cfg["training"]["type"]
    if kind == "joint":
        return [
            python,
            "-u",
            "scripts/train_joint_policy.py",
            *cfg["training"]["overrides"],
            f"hydra.run.dir={destination / 'joint'}",
            f"training.tensorboard_dir={Path(cfg['training']['local_root']) / cfg['run_id'] / 'tensorboard'}",
        ]
    if kind == "vla_dagger":
        return [
            python,
            "-u",
            "scripts/train_vla_dagger.py",
            *cfg["training"]["overrides"],
            f"hydra.run.dir={destination / 'dagger'}",
            f"data_root={Path(cfg['training']['data_root']) / cfg['run_id']}",
            f"collection.policy_name={repo / cfg['training']['joint_policy']}",
        ]
    raise ValueError(f"Unknown training type: {kind}")


def training_result(cfg, destination):
    kind = cfg["training"]["type"]
    if kind == "joint":
        (checkpoint,) = (destination / "joint/checkpoints").glob("jax_sac_final_*.pkl")
        actor = checkpoint.with_suffix(".actor.npz")
        if checkpoint.stat().st_size == 0 or actor.stat().st_size == 0:
            raise ValueError("Final joint checkpoint is empty")
        return {"training_type": kind, "checkpoint": str(checkpoint)}
    if kind == "vla_dagger":
        rounds = json.loads((destination / "dagger/rounds.json").read_text())
        final = rounds[-1]
        # S3 does not reliably expose the 'last' symlink.
        checkpoint = Path(final["checkpoint"]).parent / f"{final['training_step']:08d}"
        return {"training_type": kind, "checkpoint": str(checkpoint), "training_step": final["training_step"]}
    raise ValueError(f"Unknown training type: {kind}")


def train(cfg):
    env = dict(os.environ)
    del env["RUNPOD_API_KEY"]
    del env["TRAIN_CONFIG"]
    env.update(cfg["environment"])
    prefix = restore_environment(cfg, env)
    repo = Path(cfg["training"]["local_root"]) / "repo"
    os.chdir(repo)
    python = str(prefix / "bin/python")
    run_stage("install-project", [python, "-m", "pip", "install", "--no-build-isolation", "--editable", str(repo)], env)
    run_stage("check-dependencies", [python, "-m", "pip", "check"], env)
    destination = Path(cfg["training"]["persistent_root"]) / cfg["run_id"]
    run_stage(f"train-{cfg['training']['type']}", training_command(cfg, python, repo, destination), env)
    if cfg["training"]["type"] == "joint":
        print_time("Copying TensorBoard events to the training volume")
        shutil.copytree(Path(cfg["training"]["local_root"]) / cfg["run_id"] / "tensorboard", destination / "joint", dirs_exist_ok=True)
    (destination / "result.json").write_text(json.dumps(training_result(cfg, destination), indent=2) + "\n")


def launch(cfg):
    if cfg["training"]["type"] not in ("joint", "vla_dagger"):
        raise ValueError(f"Unknown training type: {cfg['training']['type']}")
    if cfg["pod"]["computeType"] != "GPU":
        raise ValueError("Training requires a GPU Pod")
    run_dir = create_local_run(cfg)
    key = load_api_key(cfg)
    body = dict(cfg["pod"])
    body.update(
        {
            "dockerEntrypoint": ["bash", "-lc"],
            "dockerStartCmd": [bootstrap(cfg)],
            "env": {
                **cfg["environment"],
                "TRAIN_CONFIG": json.dumps(cfg),
                "RUNPOD_API_KEY": key,
                "PUBLIC_KEY": Path(cfg["ssh_public_key_file"]).expanduser().read_text().strip(),
            },
        }
    )
    for gpu in cfg["pod"]["gpuTypeIds"]:
        body["gpuTypeIds"] = [gpu]
        print_time(f"Trying {gpu} in {body['dataCenterIds']}")
        try:
            pod = api_request(cfg, key, "POST", "/pods", body)
        except PodUnavailable:
            print_time(f"{gpu}: no instances available")
            continue
        print_time(f"Using {gpu}")
        break
    else:
        print_time("No configured GPU is available; training was not started.")
        return
    (run_dir / "pod_id").write_text(pod["id"] + "\n")
    print_time(f"Created GPU Pod {pod['id']}; run details: {run_dir}")
    print_time("Training runs independently of this terminal. Logs: https://console.runpod.io/pods")
    print_time(f"Persistent output: {cfg['training']['persistent_root']}/{cfg['run_id']}")
    ssh = wait_for_ssh(cfg, key, pod["id"])
    log_file = Path(cfg["training"]["persistent_root"]) / cfg["run_id"] / "console.log"
    command = shlex.join([*ssh, shlex.join(["tail", "-n", "100", "-F", str(log_file)])])
    print_time(f"Use this command to watch live logs (Ctrl+C stops watching, not training):\n{command}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["create", "worker"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--joint-policy", default=argparse.SUPPRESS, help="Joint checkpoint path available on the pod (VLA deployment only)")
    args = parser.parse_args()
    if "joint_policy" in vars(args) and args.command != "create":
        parser.error("--joint-policy is only supported when creating a VLA training pod")
    if args.command == "create":
        cfg = load_config(args.config, "train")
        if "joint_policy" in vars(args):
            if cfg["training"]["type"] != "vla_dagger":
                parser.error("--joint-policy requires VLA training")
            cfg["training"]["joint_policy"] = args.joint_policy
        launch(cfg)
    else:
        with args.config.open() as file:
            train(json.load(file))


if __name__ == "__main__":
    main()
