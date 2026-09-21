PORT ?= /dev/ttyACM0
ID ?= my_follower
ITERATIONS ?= 500
WARMUP ?= 20

.PHONY: install find-port test test-hardware bench lint format rollout_sim sanity_check_real sanity_check_sim rollout_real train_joint_policy train_runpod train_joint_runpod download_trained_vla download_trained_joint_policy download_pretrained

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

test-hardware:
	pytest tests/test_sensors.py -v -s --port $(PORT) --id $(ID)

lint:
	black --check src/ tests/ scripts/ benchmarks/ deployment/ diagnostics/
	ruff check src/ tests/ scripts/ benchmarks/ deployment/ diagnostics/

format:
	black src/ tests/ scripts/ benchmarks/ deployment/ diagnostics/
	ruff check --fix src/ tests/ scripts/ benchmarks/ deployment/ diagnostics/

train_joint_policy:
	python scripts/train_joint_policy.py

train_runpod:
	python deployment/runpod/train.py create --config deployment/runpod/train.toml

train_joint_runpod:
	python deployment/runpod/train.py create --config deployment/runpod/train_joint.toml

download_trained_vla:
	python deployment/runpod/download.py

download_trained_joint_policy:
	python deployment/runpod/download.py --config deployment/runpod/download_joint.toml

download_pretrained:
	python deployment/download_pretrained.py
