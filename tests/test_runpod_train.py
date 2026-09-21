import json
from pathlib import Path
import subprocess

import pytest

from deployment.runpod.common import load_config

from deployment.runpod import train


@pytest.mark.parametrize("available_index", [0, 2, 4])
def test_launch_tries_gpus_in_order(tmp_path, monkeypatch, capsys, available_index):
    cfg = load_config(Path(train.__file__).with_suffix(".toml"), "train")
    cfg.update(run_id="test-run", commit="test-commit")
    public_key = tmp_path / "key.pub"
    public_key.write_text("ssh-ed25519 test")
    cfg["ssh_public_key_file"] = str(public_key)
    monkeypatch.setattr(train, "create_local_run", lambda cfg: tmp_path)
    monkeypatch.setattr(train, "load_api_key", lambda cfg: "test-key")
    monkeypatch.setattr(train, "wait_for_ssh", lambda cfg, key, pod_id: ["ssh", "-p", "1234", "root@test-host"])
    attempts = []

    def create(cfg, key, method, endpoint, body):
        assert body["env"]["PUBLIC_KEY"] == "ssh-ed25519 test"
        assert body["ports"] == ["22/tcp"]
        attempts.append(body["gpuTypeIds"][0])
        if len(attempts) <= available_index:
            raise train.PodUnavailable("There are no instances currently available")
        return {"id": "test-pod"}

    monkeypatch.setattr(train, "api_request", create)
    train.launch(cfg)
    assert attempts == cfg["pod"]["gpuTypeIds"][: available_index + 1]
    output = capsys.readouterr().out
    for gpu in cfg["pod"]["gpuTypeIds"][:available_index]:
        assert f"{gpu}: no instances available" in output
    if available_index < 4:
        assert f"Using {attempts[-1]}" in output
        assert (tmp_path / "pod_id").read_text() == "test-pod\n"
        assert "ssh -p 1234 root@test-host" in output
        assert "tail -n 100 -F /workspace/training/test-run/console.log" in output
    else:
        assert "training was not started" in output
        assert not (tmp_path / "pod_id").exists()


@pytest.mark.parametrize("failure_stage", ["bootstrap", "clone", "training", "none"])
def test_pod_requests_termination_after_success_or_failure(tmp_path, monkeypatch, failure_stage):
    cfg = load_config(Path(train.__file__).with_suffix(".toml"), "train")
    cfg["run_id"] = "test-run"
    cfg["commit"] = "test-commit"
    cfg["training"]["local_root"] = str(tmp_path / "local")
    cfg["training"]["persistent_root"] = str(tmp_path / "volume")
    commands = tmp_path / "bin"
    commands.mkdir()
    scripts = {
        "mountpoint": "exit 0",
        "apt-get": "exit 13" if failure_stage == "bootstrap" else "exit 0",
        "git": "exit 13" if failure_stage == "clone" else f"mkdir -p '{tmp_path / 'local/repo'}'",
        "curl": 'printf "%s\\n" "$@" > "$TERMINATION_MARKER"',
        "python3": 'exit "$TRAIN_EXIT_CODE"',
        "ssh-keygen": "exit 0",
        "sshd": "exit 0",
    }
    for name, body in scripts.items():
        script = commands / name
        script.write_text("#!/bin/sh\n" + body + "\n")
        script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{commands}:/usr/bin:/bin")
    monkeypatch.setenv("TRAIN_CONFIG", json.dumps(cfg))
    monkeypatch.setenv("PUBLIC_KEY", "ssh-ed25519 test")
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setenv("RUNPOD_POD_ID", "test-pod")
    monkeypatch.setenv("TERMINATION_MARKER", str(tmp_path / "terminated"))
    monkeypatch.setenv("TRAIN_EXIT_CODE", "13" if failure_stage == "training" else "0")
    bootstrap = train.bootstrap(cfg).replace("/root/.ssh", str(tmp_path / "ssh"))
    bootstrap = bootstrap.replace("/run/sshd", str(tmp_path / "sshd")).replace("/usr/sbin/sshd", "sshd")
    result = subprocess.run(["bash", "-c", bootstrap], capture_output=True, text=True)
    expected = 0 if failure_stage == "none" else 13
    assert result.returncode == expected, result.stderr
    termination = (tmp_path / "terminated").read_text().splitlines()
    assert termination[-1] == cfg["api_url"] + "/pods/test-pod"
    assert "DELETE" in termination
    assert "Authorization: Bearer test-key" in termination
    assert (tmp_path / "volume/test-run/exit_code").read_text() == f"{expected}\n"


def test_training_uses_archive_environment_and_persistent_output(tmp_path, monkeypatch):
    cfg = load_config(Path(train.__file__).with_suffix(".toml"), "train")
    cfg["run_id"] = "test-run"
    cfg["training"]["local_root"] = str(tmp_path)
    cfg["training"]["persistent_root"] = str(tmp_path / "volume")
    destination = tmp_path / "volume/test-run"
    destination.mkdir(parents=True)
    (tmp_path / "repo").mkdir()
    monkeypatch.setenv("RUNPOD_API_KEY", "secret")
    monkeypatch.setenv("TRAIN_CONFIG", "{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "restore_environment", lambda cfg, env: tmp_path / "environment")
    stages = []
    monkeypatch.setattr(train, "run_stage", lambda name, command, env: stages.append((name, command, env)))
    monkeypatch.setattr(train, "training_result", lambda cfg, destination: {"training_type": "vla_dagger", "checkpoint": "final"})
    train.train(cfg)
    assert "--no-deps" not in stages[0][1]
    assert "--no-build-isolation" in stages[0][1]
    assert "--no-index" not in stages[0][1]
    assert "--editable" in stages[0][1]
    command = stages[-1][1]
    assert command[0] == str(tmp_path / "environment/bin/python")
    assert f"hydra.run.dir={destination / 'dagger'}" in command
    assert f"data_root={cfg['training']['data_root']}/test-run" in command
    assert f"collection.policy_name={tmp_path / 'repo' / cfg['training']['joint_policy']}" in command
    assert all("RUNPOD_API_KEY" not in env for _, _, env in stages)
    assert json.loads((destination / "result.json").read_text())["checkpoint"] == "final"


def test_joint_command_has_no_dagger_inputs(tmp_path):
    cfg = load_config((Path(train.__file__).parent / "train_joint.toml"), "train")
    command = train.training_command(cfg, "/environment/bin/python", tmp_path / "repo", tmp_path / "run")
    assert command[:3] == ["/environment/bin/python", "-u", "scripts/train_joint_policy.py"]
    assert "training.check_power_profile=false" in command
    assert f"hydra.run.dir={tmp_path / 'run/joint'}" in command
    assert not any(arg.startswith(("data_root=", "collection.")) for arg in command)


def test_joint_result_requires_both_final_checkpoint_files(tmp_path):
    cfg = {"training": {"type": "joint"}}
    checkpoints = tmp_path / "joint/checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "jax_sac_final_30000.pkl"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(FileNotFoundError):
        train.training_result(cfg, tmp_path)
    checkpoint.with_suffix(".actor.npz").write_bytes(b"actor")
    assert train.training_result(cfg, tmp_path) == {"training_type": "joint", "checkpoint": str(checkpoint)}


def test_vla_result_uses_numbered_directory(tmp_path):
    dagger = tmp_path / "dagger"
    dagger.mkdir()
    (dagger / "rounds.json").write_text(json.dumps([{"checkpoint": str(dagger / "round_001/training/checkpoints/last"), "training_step": 100}]))
    result = train.training_result({"training": {"type": "vla_dagger"}}, tmp_path)
    assert result["checkpoint"].endswith("checkpoints/00000100")


def test_unknown_training_type_fails_before_launch():
    with pytest.raises(ValueError, match="Unknown training type"):
        train.launch({"training": {"type": "unknown"}})
