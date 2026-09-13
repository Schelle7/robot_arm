from pathlib import Path
import tomllib

import pytest

from deployment.runpod import prepare_environment


@pytest.mark.parametrize("failure_stage", ["startup", "preparation"])
def test_failure_terminates_created_cpu_pod(tmp_path, monkeypatch, failure_stage):
    with (Path(prepare_environment.__file__).with_name("prepare_environment.toml")).open("rb") as file:
        cfg = tomllib.load(file)
    public_key = tmp_path / "key.pub"
    public_key.write_text("ssh-ed25519 test")
    cfg["ssh_public_key_file"] = str(public_key)
    cfg["local_run_dir"] = str(tmp_path / "runs")
    outputs = iter(["", "abc123\n", "test-api-key\n"])
    monkeypatch.setattr(prepare_environment.subprocess, "check_output", lambda *args, **kwargs: next(outputs))
    requests = []

    def api(cfg, key, method, endpoint, body):
        requests.append((method, endpoint, body))
        if method == "POST":
            return {"id": "created-cpu-pod"}

    def fail_startup(cfg, key, pod_id):
        if failure_stage == "startup":
            raise TimeoutError("SSH startup failed")
        return ["ssh", "test-pod"]

    def fail_preparation(cfg, ssh, run_dir):
        raise TimeoutError("Preparation timed out")

    monkeypatch.setattr(prepare_environment, "api_request", api)
    monkeypatch.setattr(prepare_environment, "wait_for_ssh", fail_startup)
    monkeypatch.setattr(prepare_environment, "run_preparation", fail_preparation)
    with pytest.raises(TimeoutError):
        prepare_environment.launch(cfg)

    assert requests[0][0:2] == ("POST", "/pods")
    assert requests[0][2]["computeType"] == "CPU"
    assert requests[0][2]["networkVolumeId"] == cfg["pod"]["networkVolumeId"]
    assert "RUNPOD_API_KEY" not in requests[0][2]["env"]
    assert requests[1] == ("DELETE", "/pods/created-cpu-pod", None)
    assert len(requests) == 2
    assert next((tmp_path / "runs").glob("*/pod_id")).read_text() == "created-cpu-pod\n"


def test_dirty_checkout_fails_before_creating_pod(monkeypatch):
    monkeypatch.setattr(prepare_environment.subprocess, "check_output", lambda *args, **kwargs: " M pyproject.toml\n")

    def unexpected_api(*args):
        pytest.fail("A dirty checkout must not create a paid Pod")

    monkeypatch.setattr(prepare_environment, "api_request", unexpected_api)
    with pytest.raises(RuntimeError, match="Commit and push"):
        prepare_environment.launch({"pod": {"computeType": "CPU"}})
