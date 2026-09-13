import json
import subprocess
import sys
from pathlib import Path
from prepare_environment import load_api_key, wait_for_ssh

run_dir = Path(sys.argv[1])
cfg = json.loads((run_dir / "config.json").read_text())
key = load_api_key(cfg)
subprocess.run(["ssh", "-t", *wait_for_ssh(cfg, key, (run_dir / "pod_id").read_text().strip())[1:], "btop"], check=True)
