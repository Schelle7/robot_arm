import glob
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def heading(label: str) -> None:
    print(f"\n===== {label} =====")


def run(label: str, command: list[str], timeout: int = 20) -> None:
    heading(label)
    if shutil.which(command[0]) is None:
        print(f"(not installed: {command[0]})")
        return
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"(timed out after {timeout}s)")
        return
    output = (result.stdout + result.stderr).strip()
    print(output if output else "(no output)")


def run_shell(label: str, command: str, timeout: int = 30) -> None:
    heading(label)
    try:
        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"(timed out after {timeout}s)")
        return
    output = result.stdout.strip()
    print(output if output else "(no output)")


def show_files(label: str, paths: list[str]) -> None:
    heading(label)
    for path in paths:
        try:
            print(f"{path}: {Path(path).read_text().strip()}")
        except OSError as error:
            print(f"{path}: ({error.strerror})")


def show_cpu_frequencies() -> None:
    heading("CPU current frequencies (MHz)")
    frequencies = []
    for path in sorted(glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq")):
        try:
            frequencies.append(int(Path(path).read_text()) / 1000)
        except OSError:
            continue
    if not frequencies:
        print("(no cpufreq sysfs entries)")
        return
    print(f"cores={len(frequencies)} min={min(frequencies):.0f} mean={sum(frequencies) / len(frequencies):.0f} max={max(frequencies):.0f}")


def show_tool_availability() -> None:
    heading("Diagnostic tools available")
    for tool in ("py-spy", "nvidia-smi", "sensors", "cpupower", "powerprofilesctl", "tlp-stat"):
        print(f"{tool}: {'yes' if shutil.which(tool) else 'NO'}")


def read_fan_speeds() -> str:
    result = subprocess.run(["sensors"], capture_output=True, text=True)
    speeds = re.findall(r"^fan\d+:\s+(\d+) RPM", result.stdout, flags=re.MULTILINE)
    return ",".join(speeds) if speeds else "n/a"


def read_gpu_state() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    return [field.strip() for field in result.stdout.strip().split(",")]


def read_throttle_reasons() -> str:
    result = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE"], capture_output=True, text=True)
    active = re.findall(r"^\s+(\S[^:]*?)\s+: Active$", result.stdout, flags=re.MULTILINE)
    return ",".join(name.strip() for name in active) if active else "none"


def sample_under_load(seconds: int, interval: int = 5) -> None:
    heading(f"Sampling for {seconds}s, run this while training is under way")
    print(f"{'t':>5} {'sm_mhz':>7} {'watts':>6} {'gpu_c':>6} {'util':>5} {'cpu_mhz':>8} {'fans_rpm':>12}  throttle")
    started = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - started
        if elapsed > seconds:
            return
        clock, power, temperature, utilization = read_gpu_state()
        frequencies = [int(Path(path).read_text()) / 1000 for path in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq")]
        peak_cpu = max(frequencies) if frequencies else 0
        print(f"{elapsed:5.0f} {clock:>7} {power:>6} {temperature:>6} {utilization:>5} {peak_cpu:8.0f} {read_fan_speeds():>12}  {read_throttle_reasons()}", flush=True)
        time.sleep(interval)


def main() -> None:
    if len(sys.argv) > 1:
        sample_under_load(int(sys.argv[1]))
        return

    show_tool_availability()

    run(
        "GPU summary",
        [
            "nvidia-smi",
            "--query-gpu=name,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,memory.total",
            "--format=csv",
        ],
    )
    run("GPU throttle reasons and temperature thresholds", ["nvidia-smi", "-q", "-d", "PERFORMANCE,TEMPERATURE"])
    run("GPU clock policy", ["nvidia-smi", "-q", "-d", "CLOCK"])

    show_files(
        "ACPI platform profile",
        ["/sys/firmware/acpi/platform_profile_choices", "/sys/firmware/acpi/platform_profile"],
    )
    run("Power profiles daemon", ["powerprofilesctl", "get"])
    show_files(
        "intel_pstate",
        [
            "/sys/devices/system/cpu/intel_pstate/status",
            "/sys/devices/system/cpu/intel_pstate/max_perf_pct",
            "/sys/devices/system/cpu/intel_pstate/min_perf_pct",
            "/sys/devices/system/cpu/intel_pstate/no_turbo",
        ],
    )
    show_files("CPU governor (cpu0)", ["/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"])
    show_cpu_frequencies()

    run_shell("AC / battery", "cat /sys/class/power_supply/A*/online 2>/dev/null; cat /sys/class/power_supply/BAT*/status 2>/dev/null")
    run("Sensors (fans and temperatures)", ["sensors"])

    run_shell("Enabled power and thermal services", "systemctl list-unit-files --state=enabled 2>/dev/null | grep -iE 'tlp|power|thermal|cpu|fan'")
    run_shell("nvidia-powerd (Dynamic Boost)", "systemctl is-active nvidia-powerd 2>/dev/null; systemctl is-enabled nvidia-powerd 2>/dev/null")
    run_shell("Custom systemd units mentioning cpu/power/fan", "ls -la /etc/systemd/system/ 2>/dev/null | grep -iE 'cpu|power|fan|thermal'")
    run_shell("Config files that set CPU limits", "grep -rl 'max_perf_pct\\|no_turbo\\|platform_profile' /etc/ 2>/dev/null")
    run_shell(
        "Anything that locks GPU clocks or power",
        "grep -rl 'nvidia-smi' /etc/systemd/ /etc/rc.local ~/.profile ~/.bashrc ~/.bash_aliases 2>/dev/null; "
        "grep -rh 'lgc\\|lmc\\|power-limit\\|-pl ' /etc/systemd/system/*.service 2>/dev/null",
    )
    run_shell("TLP configuration overrides", "grep -rhv '^#' /etc/tlp.conf /etc/tlp.d/*.conf 2>/dev/null | grep -v '^$'")

    print("\nRequires sudo, run separately if relevant: sudo tlp-stat -s, sudo tlp-stat -p")


if __name__ == "__main__":
    main()