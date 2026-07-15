import argparse
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

LOG_PATH = "move_to_mid.log"


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate the follower to the raw midpoint of each joint's calibrated range."
    )
    parser.add_argument("--port", type=str, required=True, help="e.g. /dev/ttyACM0")
    parser.add_argument("--id", type=str, required=True, help="calibration id, e.g. my_follower")
    parser.add_argument("--seconds", type=float, required=True, help="duration of the move")
    parser.add_argument("--hz", type=float, required=True, help="command updates per second")
    args = parser.parse_args()

    log = open(LOG_PATH, "w")

    def write(line):
        log.write(line + "\n")
        log.flush()

    follower = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    follower.connect(calibrate=False)

    try:
        bus = follower.bus
        targets = {
            motor: round((cal.range_min + cal.range_max) / 2)
            for motor, cal in bus.calibration.items()
        }
        start = bus.sync_read("Present_Position", normalize=False)

        steps = round(args.seconds * args.hz)
        dt = 1.0 / args.hz

        write(f"steps={steps} dt={dt:.4f}s hz={args.hz} seconds={args.seconds}")
        write("motor            start   ->  target    delta   ticks/step")
        for motor in targets:
            delta = targets[motor] - start[motor]
            write(
                f"  {motor:<14} {start[motor]:>5}   ->  {targets[motor]:>5}   {delta:>6}   {delta / steps:>8.2f}"
            )

        for i in range(1, steps + 1):
            alpha = i / steps
            goal = {
                motor: round(start[motor] + alpha * (targets[motor] - start[motor]))
                for motor in targets
            }
            bus.sync_write("Goal_Position", goal, normalize=False)
            present = bus.sync_read("Present_Position", normalize=False)
            cells = " ".join(f"{motor}:g{goal[motor]}/p{present[motor]}" for motor in targets)
            write(f"[{i:>4}/{steps}] a={alpha:.3f} {cells}")
            time.sleep(dt)

        time.sleep(1.0)
        reached = bus.sync_read("Present_Position", normalize=False)
        write("reached:")
        for motor in targets:
            write(f"  {motor:<14} {reached[motor]:>5}  (target {targets[motor]})")
    finally:
        follower.disconnect()
        log.close()


if __name__ == "__main__":
    main()
