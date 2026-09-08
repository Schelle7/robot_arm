import argparse
import time

import numpy as np

from robot_arm.numpy_policy import load_numpy_policy
from robot_arm.robot_schema import POLICY_OBSERVATION_NAMES


def main():
    parser = argparse.ArgumentParser(description="Benchmark exported NumPy low-level policy inference.")
    parser.add_argument("--checkpoint", required=True, help="SAC checkpoint path; its .actor.npz export is loaded.")
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--warmup", required=True, type=int)
    args = parser.parse_args()

    policy = load_numpy_policy(args.checkpoint)
    observation = {
        name: np.zeros(policy.observation_sizes[name], dtype=np.float32)
        for name in POLICY_OBSERVATION_NAMES
    }

    for _ in range(args.warmup):
        policy.predict(observation, deterministic=True)

    durations_ns = np.empty(args.iterations, dtype=np.int64)
    for iteration in range(args.iterations):
        started_ns = time.perf_counter_ns()
        policy.predict(observation, deterministic=True)
        durations_ns[iteration] = time.perf_counter_ns() - started_ns

    durations_ms = durations_ns / 1_000_000
    print(f"checkpoint: {args.checkpoint}")
    print(f"iterations: {args.iterations}")
    print(
        f"predict: mean={durations_ms.mean():.4f} ms  "
        f"median={np.median(durations_ms):.4f} ms  "
        f"p95={np.percentile(durations_ms, 95):.4f} ms  "
        f"p99={np.percentile(durations_ms, 99):.4f} ms  "
        f"max={durations_ms.max():.4f} ms"
    )
    print(f"20 Hz period used: {durations_ms.mean() / 50 * 100:.3f}%")


if __name__ == "__main__":
    main()
