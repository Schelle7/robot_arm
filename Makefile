CONDA_ENV := lerobot
RUN := conda run -n $(CONDA_ENV)
PORT ?= /dev/ttyACM0
ID ?= my_follower
SECONDS ?= 20
HZ ?= 50
ITERATIONS ?= 500
WARMUP ?= 20

.PHONY: install find-port test move-mid bench

install:
	$(RUN) pip install -e .

find-port:
	$(RUN) lerobot-find-port

test:
	$(RUN) python -m robot_arm.check_arm --port $(PORT)

move-mid:
	$(RUN) python -m robot_arm.move_to_mid --port $(PORT) --id $(ID) --seconds $(SECONDS) --hz $(HZ)

bench:
	$(RUN) python benchmarks/bench_reads.py --port $(PORT) --id $(ID) --iterations $(ITERATIONS) --warmup $(WARMUP)

bench-loop:
	$(RUN) python benchmarks/bench_loop.py --port $(PORT) --id $(ID) --iterations $(ITERATIONS) --warmup $(WARMUP)

verify:
	$(RUN) python -m pytest tests/test_sensors.py -v -s --port $(PORT) --id $(ID)
