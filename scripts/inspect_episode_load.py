import argparse

import numpy as np


def load_trace(data, frames) -> np.ndarray:
    """
    Per-step gripper load for one commanded delta. Older recordings wrote one diagnostics entry per
    joint step; newer ones write one per mid-level chunk and keep the fine trace in the dense
    trajectory, so the longer of the two is the one worth reading.
    """
    coarse = np.array([abs(data["cartesian_action_diagnostics"][frame]["duty_fraction"]) for frame in frames])
    dense = np.array([abs(step["reward_breakdown"]["duty_fraction"]) for frame in frames for step in data["dense_trajectory"][frame]])
    return dense if dense.size > coarse.size else coarse


def main():
    parser = argparse.ArgumentParser(description="Prints the recorded gripper load trace, per commanded delta.")
    parser.add_argument("episode_path")
    parser.add_argument("--samples", type=int, default=12, help="Evenly spaced load values shown per delta.")
    args = parser.parse_args()

    data = np.load(args.episode_path, allow_pickle=True)
    diagnostics = data["cartesian_action_diagnostics"]
    primitive_index = data["primitive_index"][: len(diagnostics)]

    for index in np.unique(primitive_index):
        frames = np.flatnonzero(primitive_index == index)
        delta = diagnostics[frames[0]]["commanded_delta_radians"]
        gripper = np.array([diagnostics[frame]["gripper_radians"] for frame in frames])
        loads = load_trace(data, frames)

        print(f"\ndelta {delta:.4f} rad   {len(loads)} samples   gripper {gripper[0]:.4f} -> {gripper[-1]:.4f}")
        print(f"  min {loads.min():.4f}   max {loads.max():.4f}   mean {loads.mean():.4f}")
        print(f"  first10 {loads[:10].mean():.4f}   last10 {loads[-10:].mean():.4f}   peak at sample {int(loads.argmax())}")

        step = max(1, len(loads) // args.samples)
        print("  trace " + "  ".join(f"{value:.3f}" for value in loads[::step]))


if __name__ == "__main__":
    main()
