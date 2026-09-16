PORT ?= /dev/ttyACM0
ID ?= my_follower
ITERATIONS ?= 500
WARMUP ?= 20

.PHONY: install find-port test test-hardware bench bench-loop lint format rollout_sim sanity_check_real sanity_check_sim rollout_real train_low_level train_runpod download_trained_vla download_pretrained

install:
	pip install -e .

find-port:
	lerobot-find-port

rollout_sim:
	python scripts/rollout_vla.py backend=sim

sanity_check_real:
	python scripts/rollout_waypoint.py backend=real

sanity_check_sim:
	python scripts/rollout_waypoint.py backend=sim

rollout_real:
	python scripts/rollout_vla.py backend=real

test:
	pytest tests/ --ignore=tests/test_sensors.py

bench:
	python benchmarks/bench_reads.py --port $(PORT) --id $(ID) --iterations $(ITERATIONS) --warmup $(WARMUP)

bench-loop:
	python benchmarks/bench_loop.py --port $(PORT) --id $(ID) --iterations $(ITERATIONS) --warmup $(WARMUP)

test-hardware:
	pytest tests/test_sensors.py -v -s --port $(PORT) --id $(ID)

lint:
	black --check src/ tests/ scripts/ benchmarks/ deployment/
	ruff check src/ tests/ scripts/ benchmarks/ deployment/

format:
	black src/ tests/ scripts/ benchmarks/ deployment/
	ruff check --fix src/ tests/ scripts/ benchmarks/ deployment/

train_low_level:
	python scripts/train_low_level.py

train_runpod:
	python deployment/runpod/train.py create --config deployment/runpod/train.toml

download_trained_vla:
	python deployment/runpod/download.py

download_pretrained:
	python deployment/download_pretrained.py
