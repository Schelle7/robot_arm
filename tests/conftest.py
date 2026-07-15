def pytest_addoption(parser):
    parser.addoption("--port", action="store", required=True, help="Serial port to the robot, e.g. /dev/ttyACM0")
    parser.addoption("--id", action="store", required=True, help="Calibration ID, e.g. my_follower")
