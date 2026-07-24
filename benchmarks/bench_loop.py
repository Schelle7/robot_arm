import argparse
import time
import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from robot_arm.backends.read_sensors import read_block


def time_call(fn, iterations):
    dts = np.empty(iterations)
    for i in range(iterations):
        t0 = time.perf_counter()
        fn()
        dts[i] = time.perf_counter() - t0
    return dts


def report(name, dts):
    ms = dts * 1e3
    print(
        f"{name:<22} mean={ms.mean():8.3f}ms  std={ms.std():8.3f}ms  "
        f"p95={np.percentile(ms, 95):8.3f}ms  max={ms.max():8.3f}ms  "
        f"rate={1.0 / dts.mean():7.1f}Hz"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Time a read-only loop vs a read+write control loop."
    )
    parser.add_argument("--port", type=str, required=True, help="e.g. /dev/ttyACM0")
    parser.add_argument("--id", type=str, required=True, help="calibration id")
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    args = parser.parse_args()

    follower = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    follower.connect(calibrate=False)

    try:
        bus = follower.bus

        # Read only
        for _ in range(args.warmup):
            read_block(bus)

        read_only_dts = time_call(lambda: read_block(bus), args.iterations)

        # Read + Write
        def full_control_loop():
            state = read_block(bus)
            targets = state["Present_Position"]
            bus.sync_write("Goal_Position", targets, normalize=False)

        for _ in range(args.warmup):
            full_control_loop()

        read_write_dts = time_call(full_control_loop, args.iterations)

        report("read_only", read_only_dts)
        report("read_and_write", read_write_dts)

        write_cost = read_write_dts.mean() - read_only_dts.mean()
        print(f"\nestimated write cost = {write_cost * 1000:.3f}ms")

    finally:
        follower.disconnect()


if __name__ == "__main__":
    main()
