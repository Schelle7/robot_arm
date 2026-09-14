from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


class PodUnavailable(RuntimeError):
    pass


def print_time(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def api_request(cfg, key, method, endpoint, body):
    request = urllib.request.Request(
        cfg["api_url"] + endpoint,
        data=json.dumps(body).encode() if method == "POST" else None,
        method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "robot-arm-runpod/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=cfg["request_timeout_seconds"]) as response:
            result = response.read()
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        if method == "POST" and endpoint == "/pods" and "There are no instances currently available" in response_body:
            raise PodUnavailable(response_body) from error
        print(f"Runpod {method} {endpoint}: HTTP {error.code}\n{response_body}", file=sys.stderr, flush=True)
        raise
    if method == "DELETE":
        return
    return json.loads(result)


def run_stage(name, command, env):
    started = time.monotonic()
    print_time(f"START {name}")
    subprocess.run(command, env=env, check=True)
    print_time(f"DONE {name}: elapsed {time.monotonic() - started:.1f}s")


def create_local_run(cfg):
    repository = Path(__file__).resolve().parents[2]
    status = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"], text=True)
    if status:
        raise RuntimeError(f"Commit and push the deployment changes first:\n{status}")
    cfg["commit"] = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    cfg["run_id"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = repository / cfg["local_run_dir"] / cfg["run_id"]
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    return run_dir


def load_api_key(cfg):
    return subprocess.check_output([
        "bash", "-c", 'set -e; set -a; source "$1"; printenv RUNPOD_API_KEY',
        "bash", str(Path(cfg["credentials_file"]).expanduser()),
    ], text=True).strip()
