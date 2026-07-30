import math
import numpy as np
import os
import torch
import multiprocessing as mp
import logging
import gymnasium
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from stable_baselines3 import SAC

from robot_arm.envs.factory import make_env
from robot_arm.episode_runner import EpisodeRunner
from robot_arm.policies import WaypointPolicy

log = logging.getLogger(__name__)


# We need a dummy wrapper for SAC to parse the space logic implicitly
# since we dropped gym, we define simple spaces here to build the actor layout.
class DummySpaceEnv(gymnasium.Env):
    def __init__(self, cfg):
        self.observation_space = gymnasium.spaces.Dict(
            {
                "joint_positions": gymnasium.spaces.Box(
                    low=-math.pi, high=math.pi, shape=(6,), dtype=np.float32
                ),
                "joint_velocities": gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                ),
                "start_joint_positions": gymnasium.spaces.Box(
                    low=-math.pi, high=math.pi, shape=(6,), dtype=np.float32
                ),
                "high_level_action": gymnasium.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(cfg.trajectory_length, cfg.trajectory_dim),
                    dtype=np.float32,
                ),
                "time_left": gymnasium.spaces.Box(
                    low=0.0, high=np.inf, shape=(1,), dtype=np.float32
                ),
            }
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

    # Standard gymnasium API requirements for base wrappers to avoid patching crashes
    def step(self, action):
        pass

    def reset(self, *, seed=None, options=None):
        pass

    def render(self):
        pass


# We use a custom local Replay queue to ship bulk chunks, replacing the local SB3 replay buffer
class TransitionQueueBuffer:
    def __init__(self, q):
        self.q = q
        self.chunk = []

    def add(self, obs, next_obs, action, reward, done):
        self.chunk.append((obs, next_obs, action, reward, done))

    def flush(self):
        if len(self.chunk) > 0:
            self.q.put(self.chunk)
            self.chunk = []

    # EpisodeRunner uses `is not None` logic, meaning it expects truthy checking to pass
    def __bool__(self):
        return True


def worker_process(worker_id, cfg, transition_queue, chunk_reward_queue, weights_dict_server):
    """
    Subprocess isolated execution: Initializes env, runner, and an inference-only model.
    Steps physics and places transition chunks onto the queue.
    """
    ## 1. ----------------------
    torch.set_num_threads(1)

    log.info(f"Worker {worker_id}: Initializing Simulation...")
    env = make_env(cfg)

    dummy_env_for_sac = DummySpaceEnv(cfg)
    low_level_policy = SAC(
        "MultiInputPolicy", dummy_env_for_sac, buffer_size=1, device="cpu"
    )

    high_level_policy = WaypointPolicy(
        trajectory_length=cfg.trajectory_length,
        speed=cfg.training.waypoint_speed,
    )

    worker_transition_buffer = TransitionQueueBuffer(transition_queue)

    runner = EpisodeRunner(
        cfg=cfg,
        env=env,
        low_level_policy=low_level_policy,
        high_level_policy=high_level_policy,
        training=True,
        recorder=None,
        replay_buffer=worker_transition_buffer,
        chunk_reward_queue=chunk_reward_queue,
    )

    # Continuous episodes loop
    while True:
        # Sync weights before episode starts safely via the encapsulated runner policy
        runner.low_level_policy.policy.load_state_dict(weights_dict_server["weights"])

        # We must extract the simulated box pose directly from the env driver
        # to feed the waypoint generator, because we deleted info mappings in reset.

        # Runner natively collects, chunks, calls policies, and populates `worker_transition_buffer` via `.add()`
        runner.run_episode(
            instruction="grab the box",
            generate_waypoints=True,
            lift_height=cfg.training.lift_height,
            gripper_open=cfg.training.gripper_open,
            gripper_closed=cfg.training.gripper_closed,
        )

        # Batch ship all collected physics steps to the central learner
        worker_transition_buffer.flush()


def run_distributed_training(cfg: DictConfig, device: torch.device):
    """
    Spawns worker processes to collect data using inference, while the main process
    updates a central target model and distributes updated weights.
    """
    leave_computer_working = 4  # CPUs reserved for OS/Desktop usage
    # Leave 2 for OS, and subtract 1 more for the Main Learner Process pulling transitions
    num_workers = cfg.experiment.get(
        "num_workers", max(1, mp.cpu_count() - leave_computer_working - 1)
    )

    log.info(
        f"Initializing central learner with {num_workers} parallel workers on {device}..."
    )

    # Set parallel method to spawn (required for torch/CUDA safety in multiprocessing)
    mp.set_start_method("spawn", force=True)

    weights_dict_server = mp.Manager().dict()
    transition_queue = mp.Queue(maxsize=1000)
    chunk_reward_queue = mp.Queue(maxsize=1000)

    # 1. Main Learner Model Setup
    dummy_env_for_sac = DummySpaceEnv(cfg)

    # 2. Setup central SAC agent
    model = SAC(
        "MultiInputPolicy",
        dummy_env_for_sac,
        learning_rate=cfg.training.learning_rate,
        buffer_size=cfg.training.buffer_size,
        learning_starts=cfg.training.learning_starts,
        batch_size=cfg.training.batch_size,
        tau=cfg.training.tau,
        train_freq=cfg.training.train_freq,
        gradient_steps=cfg.training.gradient_steps,
        gamma=1.0,
        verbose=1,
        device=device,
    )

    hydra_cfg = HydraConfig.get()
    from stable_baselines3.common.logger import configure

    logger = configure(hydra_cfg.runtime.output_dir, ["stdout", "csv"])
    model.set_logger(logger)

    def save_checkpoint(step_name):
        output_dir = os.path.join(hydra_cfg.runtime.output_dir, "checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        model.save(os.path.join(output_dir, f"sac_manual_step_{step_name}"))

    # Initial weights sync (must happen before spawning workers)
    weights_dict_server["weights"] = model.policy.to("cpu").state_dict()
    model.policy.to(device)

    # 3. Spawn Workers
    workers = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker_process, args=(i, cfg, transition_queue, chunk_reward_queue, weights_dict_server)
        )
        p.daemon = True
        p.start()
        workers.append(p)

    log.info("Starting central learning loop...")

    global_step = 0
    target_total_steps = cfg.training.total_training_steps

    try:
        while global_step < target_total_steps:
            # Drain up to 100 rewards per loop iteration to avoid starvation
            # for _ in range(3):
            #     if chunk_reward_queue.empty():
            #         break
            #     try:
            #         c_reward = chunk_reward_queue.get_nowait()
            #         model.logger.record("rollout/ep_rew_mean", c_reward)
            #     except Exception:
            #         break

            # 1. Blocks until worker chunks arrive
            chunk = transition_queue.get()

            # 2. Add raw transitions to central buffer
            for t_obs, t_next_obs, t_action, t_reward, t_done in chunk:
                model.replay_buffer.add(
                    t_obs, t_next_obs, t_action, t_reward, t_done, [{}]
                )
                global_step += 1

                # 3. Perform learning identically to how SB3 behaves
                if (
                    global_step > model.learning_starts
                    and global_step % model.train_freq.frequency == 0
                ):
                    model.train(
                        gradient_steps=model.gradient_steps, batch_size=model.batch_size
                    )

                    if global_step % 1000 == 0:
                        model.logger.dump(step=global_step)

                        # Sync weights occasionally during heavy training (safely to CPU)
                        cpu_state_dict = {
                            k: v.cpu() for k, v in model.policy.state_dict().items()
                        }
                        weights_dict_server["weights"] = cpu_state_dict

            # Intermediate saving logic equivalent
            if global_step % 10000 == 0:
                save_checkpoint(f"step_{global_step}")

    except KeyboardInterrupt:
        log.info("Keyboard interrupt, shutting down workers...")

    save_checkpoint(f"final_{global_step}")

    for w in workers:
        w.terminate()
