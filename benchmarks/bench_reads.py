import argparse
import time

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from robot_arm.backends.read_sensors import (
    read_registers_naive,
    read_temperature,
    read_block,
)


def time_call(fn, iterations):
    dts = np.empty(iterations)
    for i in range(iterations):
        t0 = time.perf_counter()
        fn()
        dts[i] = time.perf_counter() - t0
    return dts


def report(name, dts):
    ms = dts * 1e3
    print(f"{name:<22} mean={ms.mean():8.3f}ms  std={ms.std():8.3f}ms  " f"p95={np.percentile(ms, 95):8.3f}ms  max={ms.max():8.3f}ms  " f"rate={1.0 / dts.mean():7.1f}Hz")


def main():
    parser = argparse.ArgumentParser(description="Time reading the full feedback register block vs a single temperature read.")
    parser.add_argument("--port", type=str, required=True, help="e.g. /dev/ttyACM0")
    parser.add_argument("--id", type=str, required=True, help="calibration id, e.g. my_follower")
    parser.add_argument("--iterations", type=int, required=True, help="timed reads per function")
    parser.add_argument("--warmup", type=int, required=True, help="untimed reads before timing")
    args = parser.parse_args()

    follower = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    follower.connect(calibrate=False)

    try:
        bus = follower.bus

        for _ in range(args.warmup):
            read_registers_naive(bus)
            read_temperature(bus)
            read_block(bus)

        registers_dts = time_call(lambda: read_registers_naive(bus), args.iterations)
        temperature_dts = time_call(lambda: read_temperature(bus), args.iterations)
        block_dts = time_call(lambda: read_block(bus), args.iterations)

        n_regs = 5
        report("read_registers (5 slow)", registers_dts)
        report("read_temperature (1)", temperature_dts)
        report("read_block (5 fast)", block_dts)
        print(f"\nregisters read per call = {n_regs}")
        print(f"ratio (naive / temp) = {registers_dts.mean() / temperature_dts.mean():.2f}x")
        print(f"ratio (block / temp) = {block_dts.mean() / temperature_dts.mean():.2f}x")
    finally:
        follower.disconnect()


if __name__ == "__main__":
    main()
