import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deployment.runpod.common import print_time, api_request, run_stage, create_local_run, load_api_key


def bootstrap():
    return """set -euo pipefail
echo 'Bootstrap: installing SSH, Git and download tools'
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openssh-server git curl ca-certificates bzip2 btop
mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh
printf '%s\\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A
echo 'Bootstrap: SSH ready'
exec /usr/sbin/sshd -D -e
"""


def wait_for_ssh(cfg, key, pod_id):
    deadline = time.monotonic() + cfg["startup_timeout_seconds"]
    while time.monotonic() < deadline:
        pod = api_request(cfg, key, "GET", f"/pods/{pod_id}", None)
        if pod["desiredStatus"] != "RUNNING":
            raise RuntimeError(f"Pod entered {pod['desiredStatus']} during startup")
        # Runpod omits the address fields until network provisioning completes.
        if "publicIp" in pod and "portMappings" in pod and "22" in pod["portMappings"]:
            ssh = [
                "ssh", "-p", str(pod["portMappings"]["22"]),
                "-i", str(Path(cfg["ssh_identity_file"]).expanduser()),
                "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"ConnectTimeout={cfg['poll_seconds']}",
                "-o", f"ServerAliveInterval={cfg['poll_seconds']}", "-o", "ServerAliveCountMax=3",
                f"root@{pod['publicIp']}",
            ]
            probe = subprocess.run([*ssh, "true"], capture_output=True, text=True, timeout=cfg["request_timeout_seconds"])
            if probe.returncode == 0:
                return ssh
            print_time(f"Waiting for SSH: {probe.stderr.strip()}")
        else:
            print_time("Waiting for Pod network provisioning")
        time.sleep(cfg["poll_seconds"])
    raise TimeoutError("Pod did not become SSH-ready; inspect its Runpod bootstrap logs")


def create_preparation_directories(cfg):
    volume = Path(cfg["pod"]["volumeMountPath"])
    if not os.path.ismount(volume):
        raise RuntimeError(f"Network volume is not mounted at {volume}")
    root = Path(cfg["installation"]["local_root"])
    if root.stat().st_dev == volume.stat().st_dev:
        raise RuntimeError("Installation directory must be on local disk, outside the network volume")
    (root / "tmp").mkdir()
    destination = Path(cfg["installation"]["persistent_root"]) / cfg["run_id"]
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "config.json").write_text(json.dumps(cfg, indent=2))
    return root, destination


def installation_environment(cfg, root):
    env = dict(os.environ)
    env.update({
        "CONDA_PKGS_DIRS": str(root / "packages"),
        "PIP_CACHE_DIR": str(root / "pip-cache"),
        "TMPDIR": str(root / "tmp"),
        "HF_HOME": cfg["installation"]["hf_home"],
        "HF_HUB_CACHE": str(Path(cfg["installation"]["hf_home"]) / "hub"),
        "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0", "PYTHONUNBUFFERED": "1",
        "PATH": f"{root / 'environment/bin'}:{root / 'miniforge/bin'}:{env['PATH']}",
    })
    return env


def clone_repository(cfg, ssh):
    root = Path(cfg["installation"]["local_root"])
    repo = str(root / "repo")
    commands = [
        ["mkdir", "-p", str(root)],
        ["git", "clone", "--", cfg["installation"]["repository_url"], repo],
        ["git", "-C", repo, "checkout", "--detach", cfg["commit"]],
    ]
    print_time(f"Cloning repository and checking out {cfg['commit']}")
    command = " && ".join(shlex.join(parts) for parts in commands)
    subprocess.run([*ssh, command], check=True, timeout=cfg["installation_timeout_seconds"])


def install_conda_environment(cfg, root, env):
    install = cfg["installation"]
    installer = str(root / "miniforge.sh")
    conda_root = root / "miniforge"
    prefix = str(root / "environment")
    run_stage("download-miniforge", ["curl", "-fL", install["miniforge_url"], "-o", installer], env)
    run_stage("install-miniforge", ["bash", installer, "-b", "-p", str(conda_root)], env)
    run_stage("create-environment", [
        str(conda_root / "bin/conda"), "create", "--yes", "--channel", "conda-forge",
        "--prefix", prefix, f"python={install['python_version']}", *install["conda_packages"],
    ], env)
    run_stage("install-packer", [str(root / "environment/bin/python"), "-m", "pip", "install", install["pack_package"]], env)


def install_project(root, env):
    python = str(root / "environment/bin/python")
    # A non-editable install prevents the archive from depending on the temporary checkout.
    run_stage("install-project", [python, "-m", "pip", "install", "--progress-bar", "on", str(root / "repo")], env)
    run_stage("check-dependencies", [python, "-m", "pip", "check"], env)
    run_stage("check-imports", [python, "-c", "import torch, jax, mujoco, lerobot, transformers; print('Training libraries imported successfully')"], env)


def download_models(cfg, root, env):
    for model in cfg["models"]:
        run_stage(f"download {model['repository']}", [str(root / "environment/bin/hf"), "download", model["repository"], *model["files"]], env)


def pack_environment(root, env):
    archive = root / "environment.tar.gz"
    run_stage("pack-environment", [str(root / "environment/bin/conda-pack"), "--prefix", str(root / "environment"), "--output", str(archive)], env)
    return archive


def save_archive(archive, destination, env):
    partial = destination / "environment.tar.gz.partial"
    run_stage("save-archive", ["rsync", "--info=progress2", "--fsync", str(archive), str(partial)], env)
    partial.rename(destination / archive.name)


def prepare_environment(cfg):
    root, destination = create_preparation_directories(cfg)
    env = installation_environment(cfg, root)
    install_conda_environment(cfg, root, env)
    install_project(root, env)
    download_models(cfg, root, env)
    archive = pack_environment(root, env)
    save_archive(archive, destination, env)
    print_time(f"COMPLETE: {destination}; GPU compatibility has not been tested")


def create_cpu_pod(cfg, key):
    body = dict(cfg["pod"])
    body.update({
        "dockerEntrypoint": ["bash", "-lc"], "dockerStartCmd": [bootstrap()],
        "env": {"PUBLIC_KEY": Path(cfg["ssh_public_key_file"]).expanduser().read_text().strip()},
    })
    print_time(f"Creating CPU Pod in {body['dataCenterIds']}")
    pod = api_request(cfg, key, "POST", "/pods", body)
    return pod["id"]


def upload_file(ssh, source, destination, timeout):
    print_time(f"Uploading {source.name}")
    with source.open("rb") as file:
        subprocess.run([*ssh, f"cat > {shlex.quote(str(destination))}"], stdin=file, check=True, timeout=timeout)


def run_preparation(cfg, ssh, run_dir):
    clone_repository(cfg, ssh)
    remote_dir = Path(cfg["installation"]["upload_dir"])
    timeout = cfg["request_timeout_seconds"]
    subprocess.run([*ssh, shlex.join(["mkdir", "-p", str(remote_dir)])], check=True, timeout=timeout)
    script = Path(cfg["installation"]["local_root"]) / "repo/deployment/runpod/prepare_environment.py"
    config = run_dir / "config.json"
    upload_file(ssh, config, remote_dir / config.name, timeout)
    command = shlex.join(["python3", "-u", str(script), "worker", "--config", str(remote_dir / config.name)])
    subprocess.run([*ssh, command], check=True, timeout=cfg["installation_timeout_seconds"])


def terminate_cpu_pod(cfg, key, pod_id):
    print_time(f"Terminating CPU Pod {pod_id}; preserving the network volume")
    api_request(cfg, key, "DELETE", f"/pods/{pod_id}", None)
    print_time(f"CPU Pod {pod_id} terminated")


def launch(cfg):
    if cfg["pod"]["computeType"] != "CPU":
        raise ValueError("This preparation script only creates CPU Pods")
    run_dir = create_local_run(cfg)
    key = load_api_key(cfg)
    pod_id = create_cpu_pod(cfg, key)
    try:
        (run_dir / "pod_id").write_text(pod_id + "\n")
        print_time(f"Created CPU Pod {pod_id}; run details: {run_dir}")
        print_time("Bootstrap output: https://console.runpod.io/pods")
        ssh = wait_for_ssh(cfg, key, pod_id)
        run_preparation(cfg, ssh, run_dir)
    finally:
        terminate_cpu_pod(cfg, key, pod_id)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("create")
    start.add_argument("--config", required=True, type=Path)
    worker = commands.add_parser("worker")
    worker.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "create":
        with args.config.open("rb") as file:
            launch(tomllib.load(file))
    else:
        with args.config.open() as file:
            prepare_environment(json.load(file))


if __name__ == "__main__":
    main()
