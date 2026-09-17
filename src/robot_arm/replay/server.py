import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from uuid import uuid4

from robot_arm.robot_schema import HISTORY_FEATURE_NAMES, MOTOR_ORDER

PAGE = Path(__file__).with_name("index.html").read_bytes()


class ReplayServer:
    """
    Serves whatever the viewer last displayed. The page polls rather than being pushed to, so the
    viewer loop never waits on the browser and the MuJoCo window keeps responding on its own.
    """

    def __init__(self, port, frame_count, cartesian_hz, episode_path, branch_request_path, duty_limits, joint_hz, actions, branch_available):
        self.episode_id = uuid4().hex
        self.state = {
            "episode_id": self.episode_id,
            "sections": [],
            "warnings": [],
            "current_frame": 0,
            "current_joint_step": 0,
            "joint_step_count": 0,
            "frame_count": frame_count,
            "frame_period": 1.0 / cartesian_hz,
            "motor_names": MOTOR_ORDER,
            "duty_limits": duty_limits,
            "auto_play": False,
            "branch_available": branch_available,
            "active_primitive": "N/A",
            "policy_history": [],
            "history_end_time": 0.0,
            "joint_hz": joint_hz,
            "history_feature_names": HISTORY_FEATURE_NAMES,
        }
        self.port = port
        self.episode_path = str(Path(episode_path).resolve())
        self.branch_request_path = Path(branch_request_path).resolve()
        self.action_data = {
            "episode_id": self.episode_id,
            "motor_names": MOTOR_ORDER,
            "joint_hz": joint_hz,
            "actions": actions,
        }
        self.command_queue = queue.SimpleQueue()

    def display(
        self,
        sections,
        warnings,
        current_frame,
        current_joint_step,
        joint_step_count,
        policy_history,
        history_end_time,
        active_primitive,
        auto_play,
    ) -> None:
        self.state = {
            "episode_id": self.episode_id,
            "sections": sections,
            "warnings": list(warnings),
            "current_frame": current_frame,
            "current_joint_step": current_joint_step,
            "joint_step_count": joint_step_count,
            "frame_count": self.state["frame_count"],
            "frame_period": self.state["frame_period"],
            "motor_names": self.state["motor_names"],
            "duty_limits": self.state["duty_limits"],
            "auto_play": auto_play,
            "branch_available": self.state["branch_available"],
            "active_primitive": active_primitive,
            "policy_history": policy_history,
            "history_end_time": history_end_time,
            "joint_hz": self.state["joint_hz"],
            "history_feature_names": self.state["history_feature_names"],
        }

    def take_commands(self):
        commands = []
        while not self.command_queue.empty():
            commands.append(self.command_queue.get_nowait())
        return commands

    def start(self) -> None:
        state_of = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/state":
                    body = json.dumps(state_of.state).encode()
                    content_type = "application/json"
                elif self.path == "/actions":
                    body = json.dumps(state_of.action_data).encode()
                    content_type = "application/json"
                else:
                    body = PAGE
                    content_type = "text/html"

                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if self.headers["X-Episode-ID"] != state_of.episode_id:
                    self.send_error(409, "Episode changed; wait for the viewer to refresh")
                    return
                if self.path == "/frame":
                    content_length = int(self.headers["Content-Length"])
                    frame_index = int(self.rfile.read(content_length))
                    if frame_index < 0 or frame_index >= state_of.state["frame_count"]:
                        self.send_error(400, "Frame index out of range")
                        return
                    state_of.command_queue.put(("frame", frame_index))
                elif self.path == "/joint-step":
                    content_length = int(self.headers["Content-Length"])
                    joint_step = int(self.rfile.read(content_length))
                    if joint_step < 0 or joint_step >= state_of.state["joint_step_count"]:
                        self.send_error(400, "Joint step out of range")
                        return
                    state_of.command_queue.put(("joint_step", joint_step))
                elif self.path == "/play":
                    state_of.command_queue.put(("toggle_play", None))
                elif self.path == "/generate":
                    if not state_of.state["branch_available"]:
                        self.send_error(409, "Branch export requires recorded simulation state")
                        return
                    content_length = int(self.headers["Content-Length"])
                    browser_request = json.loads(self.rfile.read(content_length))
                    branch_request = {
                        "episode_path": state_of.episode_path,
                        "frame_index": int(browser_request["frame_index"]),
                        "duration_seconds": float(browser_request["duration_seconds"]),
                        "duties": {motor: float(browser_request["duties"][motor]) for motor in MOTOR_ORDER},
                    }
                    state_of.branch_request_path.parent.mkdir(parents=True, exist_ok=True)
                    state_of.branch_request_path.write_text(json.dumps(branch_request, indent=2) + "\n")
                else:
                    self.send_error(404)
                    return

                self.send_response(204)
                self.end_headers()

            def log_message(self, *_):
                """Silences the per-request line, which would arrive ten times a second."""

        server = HTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Replay display on http://127.0.0.1:{self.port}")
