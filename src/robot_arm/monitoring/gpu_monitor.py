import threading

import pynvml

# The software bits are routine clock management and sit active most of the time on a laptop GPU,
# so they are kept apart from the hardware bits, which only fire when the card protects itself.
HW_SLOWDOWN = pynvml.nvmlClocksThrottleReasonHwSlowdown | pynvml.nvmlClocksThrottleReasonHwThermalSlowdown | pynvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown


class GpuMonitor:
    """
    Samples NVML on its own thread so the sampling rate stays fixed when training slows down,
    which is exactly when the numbers matter.
    """

    def __init__(self, interval_seconds: float):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.max_sm_clock = pynvml.nvmlDeviceGetMaxClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
        self.power_limit_watts = pynvml.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000.0

        self.interval_seconds = interval_seconds
        self.sample: dict[str, float] = {}
        self.failure: Exception | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _read(self) -> dict[str, float]:
        sm_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
        power_watts = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle)
        return {
            "sm_clock_mhz": float(sm_clock),
            "sm_clock_fraction": sm_clock / self.max_sm_clock,
            "mem_clock_mhz": float(pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)),
            "power_watts": power_watts,
            "power_fraction": power_watts / self.power_limit_watts,
            "temperature_celsius": float(pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)),
            "utilization_percent": float(pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu),
            "sw_power_cap": float(bool(reasons & pynvml.nvmlClocksThrottleReasonSwPowerCap)),
            "sw_thermal_slowdown": float(bool(reasons & pynvml.nvmlClocksThrottleReasonSwThermalSlowdown)),
            "hw_slowdown": float(bool(reasons & HW_SLOWDOWN)),
        }

    def _run(self) -> None:
        # Raising here would print a thread traceback and leave the last sample looking live, so the
        # failure is carried back to the training loop, which is where it can be read.
        try:
            while not self.stop_event.is_set():
                self.sample = self._read()
                self.stop_event.wait(self.interval_seconds)
        except Exception as error:
            self.failure = error

    def latest(self) -> dict[str, float]:
        if self.failure is not None:
            raise self.failure
        return self.sample

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()
        pynvml.nvmlShutdown()
