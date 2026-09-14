import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deployment.runpod.common import PodUnavailable, api_request, create_local_run, load_api_key, print_time, run_stage
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


def train(cfg):
    env = dict(os.environ)
    del env["RUNPOD_API_KEY"]
    del env["TRAIN_CONFIG"]
    env.update(cfg["environment"])
    prefix = restore_environment(cfg, env)
    repo = Path(cfg["training"]["local_root"]) / "repo"
    os.chdir(repo)
    python = str(prefix / "bin/python")
    run_stage("install-project", [python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--no-index", "--editable", str(repo)], env)
    run_stage("check-dependencies", [python, "-m", "pip", "check"], env)
    destination = Path(cfg["training"]["persistent_root"]) / cfg["run_id"]
    run_stage("train-dagger", [
        python, "-u", "scripts/train_vla_dagger.py", *cfg["training"]["overrides"],
        f"hydra.run.dir={destination / 'dagger'}",
        f"data_root={Path(cfg['training']['data_root']) / cfg['run_id']}",
        f"collection.policy_name={repo / cfg['training']['joint_policy']}",
    ], env)


def launch(cfg):
    if cfg["pod"]["computeType"] != "GPU":
        raise ValueError("Training requires a GPU Pod")
    run_dir = create_local_run(cfg)
    key = load_api_key(cfg)
    body = dict(cfg["pod"])
    body.update({
        "dockerEntrypoint": ["bash", "-lc"], "dockerStartCmd": [bootstrap(cfg)],
        "env": {
            **cfg["environment"], "TRAIN_CONFIG": json.dumps(cfg), "RUNPOD_API_KEY": key,
            "PUBLIC_KEY": Path(cfg["ssh_public_key_file"]).expanduser().read_text().strip(),
        },
    })
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
    args = parser.parse_args()
    if args.command == "create":
        with args.config.open("rb") as file:
            launch(tomllib.load(file))
    else:
        with args.config.open() as file:
            train(json.load(file))


if __name__ == "__main__":
    main()
