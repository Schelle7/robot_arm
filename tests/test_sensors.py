import pytest
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from robot_arm.read_sensors import read_registers_naive, read_block

# Allowed differences between reads taken ~5ms apart
TOLERANCES = {
    "Present_Position": 5,      # A few ticks of noise/movement
    "Present_Velocity": 20,     # Velocity can fluctuate
    "Present_Load": 20,         # Load measurement is notoriously noisy
    "Present_Voltage": 1,       # Voltage (e.g. 110 = 11.0V) shouldn't jump much 
    "Present_Temperature": 1,   # Temp changes very slowly
}

@pytest.fixture
def bus(pytestconfig):
    port = pytestconfig.getoption("port")
    id_ = pytestconfig.getoption("id")
    
    follower = SO101Follower(SO101FollowerConfig(port=port, id=id_))
    follower.connect(calibrate=False)
    yield follower.bus
    follower.disconnect()

def test_block_read_matches_naive(bus):
    naive = read_registers_naive(bus)
    block = read_block(bus)
    
    errors = []
    for reg in naive:
        tol = TOLERANCES.get(reg, 0)
        
        for motor in naive[reg]:
            val_n = naive[reg][motor]
            val_b = block[reg][motor]
            diff = abs(val_n - val_b)
            
            if diff > tol:
                errors.append(f"{reg} on {motor}: naive={val_n}, block={val_b}, diff={diff} (> {tol})")
                
    assert not errors, "Found values outside expected tolerances:\n" + "\n".join(errors)
