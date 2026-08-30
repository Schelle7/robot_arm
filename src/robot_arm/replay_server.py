import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from robot_arm.robot_schema import MOTOR_ORDER

PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robot replay</title>
<style>
    :root { color-scheme: light; --ink: #17201d; --muted: #61706a; --line: #ccd5d1; --paper: #f5f7f4; --panel: #fff; --accent: #087f5b; --warn: #a33b20; }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--ink); background: repeating-linear-gradient(135deg, var(--paper) 0, var(--paper) 18px, #eef2ef 18px, #eef2ef 19px); font: 14px/1.45 "DejaVu Sans", sans-serif; }
    header { padding: 18px clamp(16px, 4vw, 48px); color: #fff; background: #17201d; display: flex; align-items: baseline; justify-content: space-between; gap: 16px; }
    h1 { margin: 0; font: 600 22px/1.2 "DejaVu Serif", serif; letter-spacing: 0; }
    #status { color: #b9c8c2; font-size: 13px; }
    main { width: min(1180px, 100%); margin: 0 auto; padding: 20px clamp(12px, 3vw, 32px) 48px; }
    .controls { display: grid; grid-template-columns: auto minmax(180px, 1fr) 84px auto; align-items: center; gap: 10px; padding: 14px 0 20px; border-bottom: 2px solid var(--ink); }
    .controls label { font-weight: 600; }
    .branch-controls { padding: 20px 0; border-bottom: 2px solid var(--ink); }
    .duty-grid { display: grid; grid-template-columns: repeat(3, minmax(150px, 1fr)); gap: 12px; }
    .duty-grid label, .generation-row label { display: grid; gap: 5px; color: var(--muted); font-size: 12px; font-weight: 600; }
    .duty-grid input, .generation-row input { width: 100%; padding: 7px 8px; border: 1px solid #899791; background: var(--panel); font: inherit; }
    .generation-row { display: grid; grid-template-columns: minmax(150px, 240px) auto 1fr; align-items: end; gap: 12px; margin-top: 14px; }
    #generationStatus { align-self: center; color: var(--muted); font-family: "DejaVu Sans Mono", monospace; }
    #dutyWarning { margin: 12px 0 0; color: var(--warn); }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    input[type="number"] { width: 84px; padding: 7px 8px; border: 1px solid #899791; background: var(--panel); font: inherit; }
    button { min-width: 78px; padding: 8px 14px; border: 1px solid #076849; background: var(--accent); color: #fff; font: inherit; font-weight: 600; cursor: pointer; }
    button:hover { background: #076849; }
    button:focus-visible, input:focus-visible { outline: 3px solid #f2b84b; outline-offset: 2px; }
    section { padding: 20px 0 4px; }
    h2 { margin: 0 0 8px; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0; }
    .table-wrap { overflow-x: auto; border-top: 1px solid var(--line); }
    table { width: 100%; border-collapse: collapse; background: rgba(255, 255, 255, 0.72); }
    th, td { padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
    th { color: var(--muted); background: #e8eeea; font-size: 12px; font-weight: 600; }
    th:first-child, td:first-child { font-weight: 600; }
    td { font-family: "DejaVu Sans Mono", monospace; font-variant-numeric: tabular-nums; }
    #warnings { margin: 18px 0 0; color: var(--warn); white-space: pre-wrap; }
    [hidden] { display: none; }
    @media (max-width: 620px) { .controls { grid-template-columns: 1fr 84px; } .controls label { grid-column: 1 / -1; } .controls button { grid-column: 1 / -1; } .duty-grid { grid-template-columns: 1fr 1fr; } .generation-row { grid-template-columns: 1fr; } header { align-items: flex-start; flex-direction: column; } }
</style>
<header><h1>Robot replay</h1><span id="status">Waiting for viewer</span></header>
<main>
<div class="controls">
    <label for="frameRange">Frame <span id="frameLabel">0 / 0</span></label>
    <input id="frameRange" type="range" min="0" max="0" value="0">
    <input id="frameNumber" type="number" min="0" max="0" value="0" aria-label="Frame number">
    <button id="playButton" type="button">Play</button>
</div>
<div class="branch-controls">
    <h2>Fixed-duty branch</h2>
    <div id="dutyGrid" class="duty-grid"></div>
    <div class="generation-row">
        <label for="durationSeconds">Duration (seconds)<input id="durationSeconds" type="number" min="0.2" step="0.2" value="1.0"></label>
        <button id="generateButton" type="button">Export branch request</button>
        <span id="generationStatus"></span>
    </div>
    <p id="dutyWarning" hidden></p>
</div>
<div id="tables"></div>
<pre id="warnings"></pre>
</main>
<script>
    const frameRange = document.getElementById("frameRange");
    const frameNumber = document.getElementById("frameNumber");
    const frameLabel = document.getElementById("frameLabel");
    const playButton = document.getElementById("playButton");
    const dutyGrid = document.getElementById("dutyGrid");
    const durationSeconds = document.getElementById("durationSeconds");
    const generateButton = document.getElementById("generateButton");
    const generationStatus = document.getElementById("generationStatus");
    const dutyWarning = document.getElementById("dutyWarning");
    const tables = document.getElementById("tables");
    const status = document.getElementById("status");
    let tableStructure = "";
    let dutyInputsCreated = false;

    function requestFrame(value) {
        frameRange.value = value;
        frameNumber.value = value;
        fetch("/frame", {method: "POST", body: value});
    }

    function togglePlay() {
        fetch("/play", {method: "POST"});
    }

    function updateDutyWarning() {
        const violations = [];
        for (const input of dutyGrid.querySelectorAll("input")) {
            const value = Number(input.value);
            const limit = Number(input.max);
            if (value < -limit || value > limit) {
                violations.push(`${input.dataset.motor}: ${value} is outside [-${limit}, ${limit}]`);
            }
        }
        dutyWarning.textContent = violations.join("; ");
        dutyWarning.hidden = violations.length === 0;
    }

    function createDutyInputs(motorNames, dutyLimits) {
        for (const motorName of motorNames) {
            const label = document.createElement("label");
            const input = document.createElement("input");
            const limit = dutyLimits[motorName];
            label.textContent = motorName.replaceAll("_", " ");
            input.type = "number";
            input.min = String(-limit);
            input.max = String(limit);
            input.step = "0.01";
            input.value = "0";
            input.dataset.motor = motorName;
            input.addEventListener("input", updateDutyWarning);
            label.appendChild(input);
            dutyGrid.appendChild(label);
        }
        dutyInputsCreated = true;
    }

    async function exportBranchRequest() {
        const duties = {};
        for (const input of dutyGrid.querySelectorAll("input")) {
            duties[input.dataset.motor] = Number(input.value);
        }
        const response = await fetch("/generate", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({frame_index: Number(frameNumber.value), duration_seconds: Number(durationSeconds.value), duties}),
        });
        if (!response.ok) {
            generationStatus.textContent = await response.text();
            return;
        }
        generationStatus.textContent = "Saved. Close replay, then run: python scripts/rollout_fixed_duty.py";
    }

    function renderTables(sections) {
        const nextStructure = JSON.stringify(sections.map(sectionData => [sectionData.title, sectionData.columns, sectionData.rows.map(row => row[0])]));
        if (nextStructure !== tableStructure) {
            tables.replaceChildren();
            for (const sectionData of sections) {
                const section = document.createElement("section");
                const heading = document.createElement("h2");
                const wrapper = document.createElement("div");
                const table = document.createElement("table");
                const head = document.createElement("thead");
                const headerRow = document.createElement("tr");
                const body = document.createElement("tbody");
                heading.textContent = sectionData.title;
                wrapper.className = "table-wrap";
                for (const column of sectionData.columns) {
                    const cell = document.createElement("th");
                    cell.textContent = column;
                    headerRow.appendChild(cell);
                }
                for (const row of sectionData.rows) {
                    const tableRow = document.createElement("tr");
                    for (const value of row) {
                        tableRow.appendChild(document.createElement("td"));
                    }
                    body.appendChild(tableRow);
                }
                head.appendChild(headerRow);
                table.append(head, body);
                wrapper.appendChild(table);
                section.append(heading, wrapper);
                tables.appendChild(section);
            }
            tableStructure = nextStructure;
        }

        const cells = tables.querySelectorAll("tbody td");
        let cellIndex = 0;
        for (const sectionData of sections) {
            for (const row of sectionData.rows) {
                for (const value of row) {
                    cells[cellIndex].textContent = value;
                    cellIndex += 1;
                }
            }
        }
    }

    frameRange.addEventListener("input", () => requestFrame(frameRange.value));
    frameNumber.addEventListener("change", () => requestFrame(frameNumber.value));
    playButton.addEventListener("click", togglePlay);
    generateButton.addEventListener("click", exportBranchRequest);
    document.addEventListener("keydown", event => {
        if (event.code === "Space" && !event.repeat) {
            event.preventDefault();
            togglePlay();
        }
    });

  setInterval(async () => {
        const state = await (await fetch("/state")).json();
        const maxFrame = state.frame_count - 1;
        frameRange.max = maxFrame;
        frameNumber.max = maxFrame;
        if (document.activeElement !== frameRange && document.activeElement !== frameNumber) {
            frameRange.value = state.current_frame;
            frameNumber.value = state.current_frame;
        }
        frameLabel.textContent = `${state.current_frame} / ${maxFrame}`;
        playButton.textContent = state.auto_play ? "Pause" : "Play";
        status.textContent = state.auto_play ? "Playing" : "Paused";
        durationSeconds.min = state.frame_period;
        durationSeconds.step = state.frame_period;
        if (!dutyInputsCreated) {
            createDutyInputs(state.motor_names, state.duty_limits);
        }
        renderTables(state.sections);
    document.getElementById("warnings").textContent = state.warnings.join("\\n");
  }, 100);
</script>
</html>
"""


class ReplayServer:
    """
    Serves whatever the viewer last displayed. The page polls rather than being pushed to, so the
    viewer loop never waits on the browser and the MuJoCo window keeps responding on its own.
    """

    def __init__(self, port, frame_count, cartesian_hz, episode_path, branch_request_path, duty_limits):
        self.state = {
            "sections": [],
            "warnings": [],
            "current_frame": 0,
            "frame_count": frame_count,
            "frame_period": 1.0 / cartesian_hz,
            "motor_names": MOTOR_ORDER,
            "duty_limits": duty_limits,
            "auto_play": False,
        }
        self.port = port
        self.episode_path = str(Path(episode_path).resolve())
        self.branch_request_path = Path(branch_request_path).resolve()
        self.command_queue = queue.SimpleQueue()

    def display(self, sections, warnings, current_frame, auto_play) -> None:
        self.state = {
            "sections": sections,
            "warnings": list(warnings),
            "current_frame": current_frame,
            "frame_count": self.state["frame_count"],
            "frame_period": self.state["frame_period"],
            "motor_names": self.state["motor_names"],
            "duty_limits": self.state["duty_limits"],
            "auto_play": auto_play,
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
                else:
                    body = PAGE.encode()
                    content_type = "text/html"

                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if self.path == "/frame":
                    content_length = int(self.headers["Content-Length"])
                    frame_index = int(self.rfile.read(content_length))
                    if frame_index < 0 or frame_index >= state_of.state["frame_count"]:
                        self.send_error(400, "Frame index out of range")
                        return
                    state_of.command_queue.put(("frame", frame_index))
                elif self.path == "/play":
                    state_of.command_queue.put(("toggle_play", None))
                elif self.path == "/generate":
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