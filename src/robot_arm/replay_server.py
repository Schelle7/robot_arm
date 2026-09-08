import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from uuid import uuid4

from robot_arm.robot_schema import HISTORY_FEATURE_NAMES, MOTOR_ORDER

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
    .low-level-controls { grid-template-columns: auto minmax(180px, 1fr) 84px; }
    .controls label { font-weight: 600; }
    .branch-controls { padding: 20px 0; border-bottom: 2px solid var(--ink); overflow-x: auto; }
    .active-primitive { display: flex; gap: 12px; padding: 16px 0; border-bottom: 2px solid var(--ink); font-size: 18px; font-weight: 700; }
    .active-primitive-label { color: var(--muted); text-transform: uppercase; }
    .duty-grid { display: grid; grid-template-columns: repeat(6, minmax(110px, 1fr)); gap: 12px; min-width: 720px; }
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
    td.pass { color: #087f5b; font-weight: 700; }
    td.fail { color: var(--warn); font-weight: 700; }
    .action-plots { border-top: 2px solid var(--ink); }
    .action-plot { padding: 12px 0; border-bottom: 1px solid var(--line); }
    .action-plot h3 { margin: 0 0 6px; color: var(--ink); font-size: 13px; font-weight: 600; }
    .action-plot canvas { display: block; width: 100%; height: 150px; background: rgba(255, 255, 255, 0.72); }
    .plot-legend { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-bottom: 6px; color: var(--muted); font: 11px "DejaVu Sans Mono", monospace; }
    .plot-legend span::before { content: ""; display: inline-block; width: 9px; height: 9px; margin-right: 5px; background: var(--series-color); }
    #warnings { margin: 18px 0 0; color: var(--warn); white-space: pre-wrap; }
    [hidden] { display: none; }
    @media (max-width: 620px) { .controls { grid-template-columns: 1fr 84px; } .controls label { grid-column: 1 / -1; } .controls button { grid-column: 1 / -1; } .generation-row { grid-template-columns: 1fr; } header { align-items: flex-start; flex-direction: column; } }
</style>
<header><h1>Robot replay</h1><span id="status">Waiting for viewer</span></header>
<main>
<div id="overviewTable"></div>
<div class="controls">
    <label for="frameRange">Frame <span id="frameLabel">0 / 0</span></label>
    <input id="frameRange" type="range" min="0" max="0" value="0">
    <input id="frameNumber" type="number" min="0" max="0" value="0" aria-label="Frame number">
    <button id="playButton" type="button">Play</button>
</div>
<div class="controls low-level-controls">
    <label for="lowLevelRange">Low-level step <span id="lowLevelLabel">N/A</span></label>
    <input id="lowLevelRange" type="range" min="0" max="0" value="0" disabled>
    <input id="lowLevelNumber" type="number" min="0" max="0" value="0" aria-label="Low-level step number" disabled>
</div>
<div class="active-primitive">
    <span class="active-primitive-label">Active primitive</span>
    <span id="activePrimitive">N/A</span>
</div>
<div id="tables"></div>
<div id="branchControls" class="branch-controls">
    <h2>Fixed policy-action branch</h2>
    <div id="dutyGrid" class="duty-grid"></div>
    <div class="generation-row">
        <label for="durationSeconds">Seconds<input id="durationSeconds" type="number" min="0.2" step="0.2" value="1.0"></label>
        <button id="generateButton" type="button">Export branch request</button>
        <span id="generationStatus"></span>
    </div>
    <p id="dutyWarning" hidden></p>
</div>
<section id="historySection" hidden>
    <h2>Policy history</h2>
    <div id="historyPlots" class="action-plots"></div>
</section>
<section>
    <h2>Policy actions</h2>
    <div id="actionPlots" class="action-plots"></div>
</section>
<pre id="warnings"></pre>
</main>
<script>
    const frameRange = document.getElementById("frameRange");
    const frameNumber = document.getElementById("frameNumber");
    const frameLabel = document.getElementById("frameLabel");
    const playButton = document.getElementById("playButton");
    const lowLevelRange = document.getElementById("lowLevelRange");
    const lowLevelNumber = document.getElementById("lowLevelNumber");
    const lowLevelLabel = document.getElementById("lowLevelLabel");
    const activePrimitive = document.getElementById("activePrimitive");
    const dutyGrid = document.getElementById("dutyGrid");
    const durationSeconds = document.getElementById("durationSeconds");
    const generateButton = document.getElementById("generateButton");
    const generationStatus = document.getElementById("generationStatus");
    const dutyWarning = document.getElementById("dutyWarning");
    const branchControls = document.getElementById("branchControls");
    const overviewTable = document.getElementById("overviewTable");
    const tables = document.getElementById("tables");
    const historySection = document.getElementById("historySection");
    const historyPlots = document.getElementById("historyPlots");
    const actionPlots = document.getElementById("actionPlots");
    const status = document.getElementById("status");
    let overviewStructure = "";
    let tableStructure = "";
    let dutyInputsCreated = false;
    let episodeId = null;
    let polling = false;
    const plotColors = ["#087f5b", "#9c36b5", "#e67700", "#1971c2", "#c92a2a", "#5f3dc4"];
    const historyGroups = [
        ["Policy action", 0, "policy_action_"],
        ["Applied duty", 6, "applied_duty_"],
        ["Joint velocity (normalized)", 12, "joint_velocity_"],
        ["TCP velocity (normalized)", 18, "tcp_velocity_"],
    ];

    function requestFrame(value) {
        frameRange.value = value;
        frameNumber.value = value;
        fetch("/frame", {method: "POST", headers: {"X-Episode-ID": episodeId}, body: value});
    }

    function togglePlay() {
        fetch("/play", {method: "POST", headers: {"X-Episode-ID": episodeId}});
    }

    function requestLowLevelStep(value) {
        lowLevelRange.value = value;
        lowLevelNumber.value = value;
        fetch("/low-level-step", {method: "POST", headers: {"X-Episode-ID": episodeId}, body: value});
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
        const requestedEpisode = episodeId;
        const duties = {};
        for (const input of dutyGrid.querySelectorAll("input")) {
            duties[input.dataset.motor] = Number(input.value);
        }
        const response = await fetch("/generate", {
            method: "POST",
            headers: {"Content-Type": "application/json", "X-Episode-ID": requestedEpisode},
            body: JSON.stringify({frame_index: Number(frameNumber.value), duration_seconds: Number(durationSeconds.value), duties}),
        });
        const message = response.ok ? "Branch request saved." : await response.text();
        if (episodeId === requestedEpisode) generationStatus.textContent = message;
    }

    function renderTableContainer(container, sections, currentStructure) {
        const nextStructure = JSON.stringify(sections.map(sectionData => [sectionData.title, sectionData.columns, sectionData.rows.map(row => row[0])]));
        if (nextStructure !== currentStructure) {
            container.replaceChildren();
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
                container.appendChild(section);
            }
        }

        const sectionElements = container.querySelectorAll(":scope > section");
        for (let sectionIndex = 0; sectionIndex < sections.length; sectionIndex += 1) {
            const sectionData = sections[sectionIndex];
            const tableRows = sectionElements[sectionIndex].querySelectorAll("tbody tr");
            for (let rowIndex = 0; rowIndex < sectionData.rows.length; rowIndex += 1) {
                const row = sectionData.rows[rowIndex];
                const cells = tableRows[rowIndex].querySelectorAll("td");
                for (let cellIndex = 0; cellIndex < row.length; cellIndex += 1) {
                    cells[cellIndex].textContent = row[cellIndex];
                    cells[cellIndex].className = "";
                }
                if (sectionData.title === "Primitive") {
                    if (row[0] === "Completes active primitive") {
                        cells[1].className = row[1] === "True" ? "pass" : "fail";
                    } else {
                        cells[1].className = Number(row[1]) <= Number(row[2]) ? "pass" : "fail";
                    }
                }
            }
        }
        return nextStructure;
    }

    function renderTables(sections) {
        overviewStructure = renderTableContainer(overviewTable, sections.slice(0, 1), overviewStructure);
        tableStructure = renderTableContainer(tables, sections.slice(1), tableStructure);
    }

    function drawActionPlot(canvas, values, jointHz) {
        const pixelRatio = window.devicePixelRatio || 1;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        canvas.width = Math.round(width * pixelRatio);
        canvas.height = Math.round(height * pixelRatio);
        const context = canvas.getContext("2d");
        context.scale(pixelRatio, pixelRatio);

        const left = 38;
        const right = 12;
        const top = 10;
        const bottom = 24;
        const plotWidth = width - left - right;
        const plotHeight = height - top - bottom;
        const x = index => left + (values.length === 1 ? 0 : index * plotWidth / (values.length - 1));
        const y = value => top + (1 - value) * plotHeight / 2;

        context.strokeStyle = "#ccd5d1";
        context.lineWidth = 1;
        for (const value of [-1, 0, 1]) {
            context.beginPath();
            context.moveTo(left, y(value));
            context.lineTo(width - right, y(value));
            context.stroke();
            context.fillStyle = "#61706a";
            context.font = '11px "DejaVu Sans Mono", monospace';
            context.textAlign = "right";
            context.textBaseline = "middle";
            context.fillText(String(value), left - 6, y(value));
        }

        const durationSeconds = Math.max(values.length - 1, 0) / jointHz;
        context.fillStyle = "#61706a";
        context.textBaseline = "bottom";
        context.textAlign = "left";
        context.fillText("0 s", left, height);
        context.textAlign = "right";
        context.fillText(`${durationSeconds.toFixed(2)} s`, width - right, height);

        context.fillStyle = "#087f5b";
        for (let index = 0; index < values.length; index += 1) {
            context.beginPath();
            context.arc(x(index), y(values[index]), 1.75, 0, 2 * Math.PI);
            context.fill();
        }
    }

    function drawHistoryPlot(canvas, history, startIndex, endTime, jointHz) {
        const pixelRatio = window.devicePixelRatio || 1;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        canvas.width = Math.round(width * pixelRatio);
        canvas.height = Math.round(height * pixelRatio);
        const context = canvas.getContext("2d");
        context.scale(pixelRatio, pixelRatio);

        const left = 48;
        const right = 12;
        const top = 10;
        const bottom = 24;
        const plotWidth = width - left - right;
        const plotHeight = height - top - bottom;
        const values = history.flatMap(sample => sample.slice(startIndex, startIndex + 6));
        const magnitude = Math.max(1.0, ...values.map(value => Math.abs(value)));
        const x = index => left + index * plotWidth / (history.length - 1);
        const y = value => top + (magnitude - value) * plotHeight / (2 * magnitude);

        context.strokeStyle = "#ccd5d1";
        context.lineWidth = 1;
        for (const value of [-magnitude, 0, magnitude]) {
            context.beginPath();
            context.moveTo(left, y(value));
            context.lineTo(width - right, y(value));
            context.stroke();
            context.fillStyle = "#61706a";
            context.font = '11px "DejaVu Sans Mono", monospace';
            context.textAlign = "right";
            context.textBaseline = "middle";
            context.fillText(value.toFixed(2), left - 6, y(value));
        }

        for (let seriesIndex = 0; seriesIndex < 6; seriesIndex += 1) {
            context.strokeStyle = plotColors[seriesIndex];
            context.lineWidth = 1.5;
            context.beginPath();
            for (let historyIndex = 0; historyIndex < history.length; historyIndex += 1) {
                const pointX = x(historyIndex);
                const pointY = y(history[historyIndex][startIndex + seriesIndex]);
                if (historyIndex === 0) {
                    context.moveTo(pointX, pointY);
                } else {
                    context.lineTo(pointX, pointY);
                }
            }
            context.stroke();
            context.fillStyle = plotColors[seriesIndex];
            for (let historyIndex = 0; historyIndex < history.length; historyIndex += 1) {
                context.beginPath();
                context.arc(x(historyIndex), y(history[historyIndex][startIndex + seriesIndex]), 2.5, 0, 2 * Math.PI);
                context.fill();
            }
        }

        context.fillStyle = "#61706a";
        context.textBaseline = "bottom";
        context.textAlign = "left";
        context.fillText(`${(endTime - (history.length - 1) / jointHz).toFixed(2)} s`, left, height);
        context.textAlign = "right";
        context.fillText(`${endTime.toFixed(2)} s`, width - right, height);
        context.textAlign = "center";
        context.fillText("Episode time (negative = reset padding)", left + plotWidth / 2, height);
    }

    function renderHistoryPlots(history, featureNames, endTime, jointHz) {
        historySection.hidden = history.length === 0;
        if (history.length === 0) {
            return;
        }
        if (historyPlots.childElementCount === 0) {
            for (const [title, startIndex, featurePrefix] of historyGroups) {
                const plot = document.createElement("div");
                const heading = document.createElement("h3");
                const legend = document.createElement("div");
                const canvas = document.createElement("canvas");
                plot.className = "action-plot";
                heading.textContent = title;
                legend.className = "plot-legend";
                for (let seriesIndex = 0; seriesIndex < 6; seriesIndex += 1) {
                    const item = document.createElement("span");
                    item.textContent = featureNames[startIndex + seriesIndex].replace(featurePrefix, "").replaceAll("_", " ");
                    item.style.setProperty("--series-color", plotColors[seriesIndex]);
                    legend.appendChild(item);
                }
                canvas.dataset.startIndex = startIndex;
                plot.append(heading, legend, canvas);
                historyPlots.appendChild(plot);
            }
        }
        for (const canvas of historyPlots.querySelectorAll("canvas")) {
            drawHistoryPlot(canvas, history, Number(canvas.dataset.startIndex), endTime, jointHz);
        }
    }

    function createActionPlots(actionData) {
        for (let motorIndex = 0; motorIndex < actionData.motor_names.length; motorIndex += 1) {
            const plot = document.createElement("div");
            const heading = document.createElement("h3");
            const canvas = document.createElement("canvas");
            const values = actionData.actions.map(action => action[motorIndex]);
            plot.className = "action-plot";
            heading.textContent = actionData.motor_names[motorIndex].replaceAll("_", " ");
            plot.append(heading, canvas);
            actionPlots.appendChild(plot);
            drawActionPlot(canvas, values, actionData.joint_hz);
        }
    }

    frameRange.addEventListener("input", () => requestFrame(frameRange.value));
    frameNumber.addEventListener("change", () => requestFrame(frameNumber.value));
    lowLevelRange.addEventListener("input", () => requestLowLevelStep(lowLevelRange.value));
    lowLevelNumber.addEventListener("change", () => requestLowLevelStep(lowLevelNumber.value));
    playButton.addEventListener("click", togglePlay);
    generateButton.addEventListener("click", exportBranchRequest);
    document.addEventListener("keydown", event => {
        if (event.code === "Space" && !event.repeat) {
            event.preventDefault();
            togglePlay();
        }
    });
    function resetEpisode(state, actionData) {
        overviewStructure = "";
        tableStructure = "";
        dutyInputsCreated = false;
        for (const container of [overviewTable, tables, dutyGrid, historyPlots, actionPlots]) {
            container.replaceChildren();
        }
        generationStatus.textContent = "";
        dutyWarning.textContent = "";
        dutyWarning.hidden = true;
        historySection.hidden = true;
        durationSeconds.value = state.frame_period;
        for (const input of [frameRange, frameNumber, lowLevelRange, lowLevelNumber]) {
            input.blur();
            input.value = 0;
        }
        createActionPlots(actionData);
        episodeId = state.episode_id;
    }

    async function pollState() {
        const state = await (await fetch("/state", {cache: "no-store"})).json();
        if (state.episode_id !== episodeId) {
            const actionData = await (await fetch("/actions", {cache: "no-store"})).json();
            if (actionData.episode_id !== state.episode_id) return;
            resetEpisode(state, actionData);
        }
        const maxFrame = state.frame_count - 1;
        frameRange.max = maxFrame;
        frameNumber.max = maxFrame;
        if (document.activeElement !== frameRange && document.activeElement !== frameNumber) {
            frameRange.value = state.current_frame;
            frameNumber.value = state.current_frame;
        }
        frameLabel.textContent = `${state.current_frame} / ${maxFrame}`;
        const maxLowLevelStep = Math.max(state.low_level_step_count - 1, 0);
        lowLevelRange.max = maxLowLevelStep;
        lowLevelNumber.max = maxLowLevelStep;
        lowLevelRange.disabled = state.low_level_step_count === 0;
        lowLevelNumber.disabled = state.low_level_step_count === 0;
        if (document.activeElement !== lowLevelRange && document.activeElement !== lowLevelNumber) {
            lowLevelRange.value = state.current_low_level_step;
            lowLevelNumber.value = state.current_low_level_step;
        }
        lowLevelLabel.textContent = state.low_level_step_count === 0 ? "N/A" : `${state.current_low_level_step + 1} / ${state.low_level_step_count}`;
        activePrimitive.textContent = state.active_primitive;
        renderHistoryPlots(state.policy_history, state.history_feature_names, state.history_end_time, state.joint_hz);
        playButton.textContent = state.auto_play ? "Pause" : "Play";
        status.textContent = state.auto_play ? "Playing" : "Paused";
        branchControls.hidden = !state.branch_available;
        durationSeconds.min = state.frame_period;
        durationSeconds.step = state.frame_period;
        if (!dutyInputsCreated) {
            createDutyInputs(state.motor_names, state.duty_limits);
        }
        renderTables(state.sections);
    document.getElementById("warnings").textContent = state.warnings.join("\\n");
    }

    setInterval(async () => {
        if (polling) return;
        polling = true;
        try {
            await pollState();
        } finally {
            polling = false;
        }
    }, 100);
</script>
</html>
"""


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
            "current_low_level_step": 0,
            "low_level_step_count": 0,
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
        current_low_level_step,
        low_level_step_count,
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
            "current_low_level_step": current_low_level_step,
            "low_level_step_count": low_level_step_count,
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
                    body = PAGE.encode()
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
                elif self.path == "/low-level-step":
                    content_length = int(self.headers["Content-Length"])
                    low_level_step = int(self.rfile.read(content_length))
                    if low_level_step < 0 or low_level_step >= state_of.state["low_level_step_count"]:
                        self.send_error(400, "Low-level step out of range")
                        return
                    state_of.command_queue.put(("low_level_step", low_level_step))
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
