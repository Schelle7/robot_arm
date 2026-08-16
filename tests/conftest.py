def pytest_addoption(parser):
    parser.addoption(
        "--port",
        action="store",
        default=None,
        help="Serial port to the robot, e.g. /dev/ttyACM0",
    )
    parser.addoption("--id", action="store", default=None, help="Calibration ID, e.g. my_follower")
