import argparse

from lerobot.motors.feetech import FeetechMotorsBus


def main():
    parser = argparse.ArgumentParser(
        description="Scan a serial port for Feetech motor ids across all baudrates."
    )
    parser.add_argument("--port", type=str, required=True, help="e.g. /dev/ttyACM0")
    args = parser.parse_args()

    found = FeetechMotorsBus.scan_port(args.port)

    if not found:
        print(f"\nNo motors responded on {args.port}.")
        print("Either nothing is connected/powered, or several motors share the")
        print("same id (default 1) and collide on the bus. Set unique ids with:")
        print(f"  lerobot-setup-motors --robot.type=so101_follower --robot.port={args.port}")
        return

    print(f"\nMotors found on {args.port} (baudrate -> ids):")
    for baudrate, ids in found.items():
        print(f"  {baudrate}: {sorted(ids)}")

    all_ids = sorted({i for ids in found.values() for i in ids})
    if all_ids == [1, 2, 3, 4, 5, 6]:
        print("\nAll 6 ids present and unique. Motors are configured. Skip lerobot-setup-motors.")
    else:
        print(f"\nGot ids {all_ids}, expected [1, 2, 3, 4, 5, 6].")
        print("Motors are not fully configured. Run lerobot-setup-motors (one motor at a time).")


if __name__ == "__main__":
    main()
