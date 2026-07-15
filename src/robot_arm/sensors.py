FEEDBACK_REGISTERS = (
    "Present_Position",
    "Present_Velocity",
    "Present_Load",
    "Present_Voltage",
    "Present_Temperature",
)


def read_registers_naive(bus):
    """The naive way: 5 separate round-trips over the serial bus."""
    return {reg: bus.sync_read(reg, normalize=False) for reg in FEEDBACK_REGISTERS}


def read_temperature(bus):
    """Unit test: 1 single round trip."""
    return bus.sync_read("Present_Temperature", normalize=False)


def read_block(bus):
    """
    The smart way: a single 8-byte block read (addr 56 to 63) per cycle.
    1 round-trip for all 5 values.
    """
    motor_ids = [m.id for m in bus.motors.values()]
    
    # Address 56 is Present_Position. 8 bytes gets us up to Present_Temperature.
    bus._setup_sync_reader(motor_ids, 56, 8)
    
    comm = bus.sync_reader.txRxPacket()
    if not bus._is_comm_success(comm):
        raise ConnectionError(f"Block read failed: {bus.packet_handler.getTxRxResult(comm)}")
        
    results = {reg: {} for reg in FEEDBACK_REGISTERS}
    for name, motor in bus.motors.items():
        i = motor.id
        if not bus.sync_reader.isAvailable(i, 56, 8):
            continue
            
        # Extract from the already-fetched buffer
        pos  = bus.sync_reader.getData(i, 56, 2)
        vel  = bus.sync_reader.getData(i, 58, 2)
        load = bus.sync_reader.getData(i, 60, 2)
        volt = bus.sync_reader.getData(i, 62, 1)
        temp = bus.sync_reader.getData(i, 63, 1)
        
        # Velocity and Load are sign-magnitude encoded, handle decoding 
        vel  = bus._decode_sign("Present_Velocity", {i: vel})[i]
        load = bus._decode_sign("Present_Load", {i: load})[i]
        
        results["Present_Position"][name] = pos
        results["Present_Velocity"][name] = vel
        results["Present_Load"][name] = load
        results["Present_Voltage"][name] = volt
        results["Present_Temperature"][name] = temp
        
    return results
