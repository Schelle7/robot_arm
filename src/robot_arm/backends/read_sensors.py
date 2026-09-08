import time


FEEDBACK_REGISTERS = (
    "Present_Position",
    "Present_Velocity",
    "Present_Load",
    "Present_Voltage",
    "Present_Temperature",
    "Present_Current",
)

# Firmware settings that shape how a commanded position becomes motion, as (address, byte width).
# LeRobot writes several of these on every connect and nothing has ever recorded what they were.
CONFIGURATION_REGISTERS = {
    "Max_Torque_Limit": (16, 2),
    "P_Coefficient": (21, 1),
    "D_Coefficient": (22, 1),
    "I_Coefficient": (23, 1),
    "Min_Startup_Force": (24, 2),
    "CW_Dead_Zone": (26, 1),
    "CCW_Dead_Zone": (27, 1),
    "Protection_Current": (28, 2),
    "Protection_Torque": (34, 1),
    "Protection_Time": (35, 1),
    "Overload_Torque": (36, 1),
}


def read_registers_naive(bus):
    """The naive way: 5 separate round-trips over the serial bus."""
    return {reg: bus.sync_read(reg, normalize=False) for reg in FEEDBACK_REGISTERS}


def read_temperature(bus):
    """Unit test: 1 single round trip."""
    return bus.sync_read("Present_Temperature", normalize=False)


def read_configuration(bus):
    """
    One block read of addresses 16 to 36, covering the servo's control gains and its protection
    limits. These never change while running, so this is meant to be called once and logged: the
    numbers decide what a commanded position delta actually does, and are otherwise invisible.
    """
    motor_ids = [m.id for m in bus.motors.values()]
    start_address = min(address for address, _ in CONFIGURATION_REGISTERS.values())
    end_address = max(address + width for address, width in CONFIGURATION_REGISTERS.values())
    length = end_address - start_address

    bus._setup_sync_reader(motor_ids, start_address, length)
    comm = bus.sync_reader.txRxPacket()
    if not bus._is_comm_success(comm):
        raise ConnectionError(f"Configuration read failed: {bus.packet_handler.getTxRxResult(comm)}")

    results = {register: {} for register in CONFIGURATION_REGISTERS}
    for name, motor in bus.motors.items():
        if not bus.sync_reader.isAvailable(motor.id, start_address, length):
            continue
        for register, (address, width) in CONFIGURATION_REGISTERS.items():
            results[register][name] = bus.sync_reader.getData(motor.id, address, width)

    return results


def read_block(bus):
    """
    The smart way: a single 15-byte block read (addr 56 to 70) per cycle.
    1 round-trip for all 6 values.
    """
    motor_ids = [m.id for m in bus.motors.values()]

    # Address 56 is Present_Position. 15 bytes gets us through Present_Current.
    bus._setup_sync_reader(motor_ids, 56, 15)

    read_started_ns = time.perf_counter_ns()
    comm = bus.sync_reader.txRxPacket()
    read_completed_ns = time.perf_counter_ns()
    if not bus._is_comm_success(comm):
        raise ConnectionError(f"Block read failed: {bus.packet_handler.getTxRxResult(comm)}")

    results = {reg: {} for reg in FEEDBACK_REGISTERS}
    results["read_started_ns"] = read_started_ns
    results["read_completed_ns"] = read_completed_ns
    results["sample_time_ns"] = (read_started_ns + read_completed_ns) // 2
    for name, motor in bus.motors.items():
        i = motor.id
        if not bus.sync_reader.isAvailable(i, 56, 15):
            raise ConnectionError(f"Block read failed: missing feedback for motor {name!r} (ID {i})")

        # Extract from the already-fetched buffer
        pos = bus.sync_reader.getData(i, 56, 2)
        vel = bus.sync_reader.getData(i, 58, 2)
        load = bus.sync_reader.getData(i, 60, 2)
        volt = bus.sync_reader.getData(i, 62, 1)
        temp = bus.sync_reader.getData(i, 63, 1)
        current = bus.sync_reader.getData(i, 69, 2)

        # Match bus.sync_read(normalize=False): position is signed too (bit 15).
        pos = bus._decode_sign("Present_Position", {i: pos})[i]
        vel = bus._decode_sign("Present_Velocity", {i: vel})[i]
        load = bus._decode_sign("Present_Load", {i: load})[i]

        results["Present_Position"][name] = pos
        results["Present_Velocity"][name] = vel
        results["Present_Load"][name] = load
        results["Present_Voltage"][name] = volt
        results["Present_Temperature"][name] = temp
        results["Present_Current"][name] = current

    return results
