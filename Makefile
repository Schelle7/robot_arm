PORT ?= /dev/ttyACM0
ID ?= my_follower
ITERATIONS ?= 500
WARMUP ?= 20
CAMERA ?= wrist_camera

.PHONY: install find-port test test-hardware bench lint format rollout_sim try_joint_policy_real try_joint_policy_sim rollout_real train_joint_policy train_runpod train_joint_runpod download_trained_vla download_trained_joint_policy download_pretrained

install:
	pip install -e .

.PHONY: test_camera
test_camera:
	python diagnostics/capture_camera.py camera_name=$(CAMERA)

find-port:
	lerobot-find-port

rollout_sim:
	python scripts/rollout_vla.py arm_type=sim

try_joint_policy_real:
	python scripts/rollout_waypoint.py --config-name rollout_real

try_joint_policy_sim:
	python scripts/rollout_waypoint.py arm_type=sim

rollout_real:
	python scripts/rollout_vla.py arm_type=real

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
	python deployment/runpod/train.py create --config deployment/runpod/train.toml $(if $(JOINT_POLICY),--joint-policy '$(JOINT_POLICY)')

train_joint_runpod:
	python deployment/runpod/train.py create --config deployment/runpod/train_joint.toml

download_trained_vla:
	python deployment/runpod/download.py

download_trained_joint_policy:
	python deployment/runpod/download.py --config deployment/runpod/download_joint.toml

download_pretrained:
	python deployment/download_pretrained.py
