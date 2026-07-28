CONDA_ENV := lerobot
RUN := conda run -n $(CONDA_ENV)
PORT ?= /dev/ttyACM0
ID ?= my_follower
SECONDS ?= 20
HZ ?= 50
ITERATIONS ?= 500
WARMUP ?= 20

.PHONY: install find-port test move-mid bench train rollout_sim sanity_check_real rollout_real

install:
	$(RUN) pip install -e .

find-port:
	$(RUN) lerobot-find-port

train:
	$(RUN) python scripts/train_low_level.py

rollout_sim:
	$(RUN) python scripts/rollout_vla.py backend=sim

sanity_check_real:
	$(RUN) python scripts/rollout_waypoint.py backend=real

rollout_real:
	$(RUN) python scripts/rollout_vla.py backend=real

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

lint:
	$(RUN) python -m black --target-version py312 src/ tests/ scripts/ benchmarks/
	$(RUN) python -m ruff check --fix src/ tests/ scripts/ benchmarks/
