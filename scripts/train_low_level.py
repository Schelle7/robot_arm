
# Dummy setup script to show manual buffer transition assignment
# This is a skeleton of how we will manually stuff transitions into SB3


def manual_train_loop():
    # 1. Init your RobotEnv & Models
    # env = ...
    # model = SAC("MultiInputPolicy", env, gamma=1.0, ...)
    # (We use gamma=1.0 because our task is finite horizon, strictly bounded by the chunk)

    # 2. Extract the buffer
    # replay_buffer = model.replay_buffer

    # 3. Inside the physics loop:
    # for step_idx in range(self.skip_frames):
    #     time_left = self.skip_frames - step_idx - 1
    #
    #     # Action selection
    #     action, _ = model.predict(rl_obs)
    #
    #     # Step physics
    #     next_obs, reward, env_terminated, env_truncated, info = env.step(action)
    #
    #     # CRITICAL: We enforce termination exactly at chunk completion to sever the bootstrap.
    #     # A chunk finishing is a mathematical termination (value = 0 afterwards),
    #     # not an artificial truncation (which would cause SB3 to estimate future values).
    #     chunk_terminated = (time_left == 0)
    #
    #     # True termination occurs if EITHER the physics crash/finish OR the chunk naturally completes.
    #     actual_terminated = env_terminated or chunk_terminated
    #
    #     # Add to SB3 buffer directly
    #     replay_buffer.add(
    #         rl_obs,
    #         next_rl_obs,
    #         action,
    #         reward,
    #         actual_terminated,
    #         [info]
    #     )
    #
    #     rl_obs = next_rl_obs
    pass
