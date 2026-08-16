import math
import queue
import numpy as np
import os
import torch
import multiprocessing as mp
import logging
import gymnasium
from collections import deque
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class
from stable_baselines3 import SAC
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from robot_arm.envs.factory import make_env
from robot_arm.episode_runner import EpisodeRunner
from robot_arm.model_snapshot import snapshot_model_files
from robot_arm.policies import ScriptedCartesianPolicy
from robot_arm.primitive_policy import ScriptedPrimitiveGeneratorPolicy

log = logging.getLogger(__name__)


def build_sac_policy_kwargs(cfg: DictConfig) -> dict:
    features_extractor_cfg = cfg.policy.features_extractor
    return {
        "features_extractor_class": get_class(features_extractor_cfg.class_path),
        "features_extractor_kwargs": {
            "hidden_dims": list(features_extractor_cfg.hidden_dims),
        },
        "net_arch": list(cfg.policy.net_arch),
    }


# We need a dummy wrapper for SAC to parse the space logic implicitly
# since we dropped gym, we define simple spaces here to build the actor layout.
class DummySpaceEnv(gymnasium.Env):
    def __init__(self, cfg):
        self.observation_space = gymnasium.spaces.Dict(
            {
                "joint_positions": gymnasium.spaces.Box(low=-math.pi, high=math.pi, shape=(6,), dtype=np.float32),
                "joint_velocities": gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
                "start_joint_positions": gymnasium.spaces.Box(low=-math.pi, high=math.pi, shape=(6,), dtype=np.float32),
                "cartesian_action_path": gymnasium.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(cfg.waypoint.trajectory_length, cfg.waypoint.trajectory_dim),
                    dtype=np.float32,
                ),
                "time_left": gymnasium.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

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


class MetricsQueueBuffer:
    def __init__(self, q):
        self.q = q
        self.chunk_metrics = []

    def add(self, metric_dict):
        self.chunk_metrics.append(metric_dict)

    def flush(self):
        if len(self.chunk_metrics) > 0:
            self.q.put(self.chunk_metrics)
            self.chunk_metrics = []


def create_training_queues(cfg):
    transition_queue = mp.Queue(maxsize=1000)
    metrics_queue = mp.Queue(maxsize=1000)
    worker_queues = [mp.Queue(maxsize=1) for _ in range(cfg.training.num_workers)]
    return transition_queue, metrics_queue, worker_queues


def create_central_sac_model(cfg, device):
    dummy_env_for_sac = DummySpaceEnv(cfg)
    if "continue_from" in cfg:
        model = SAC.load(cfg.continue_from, env=dummy_env_for_sac, device=device)
        model.learning_starts = cfg.training.learning_starts
        return model

    policy_kwargs = build_sac_policy_kwargs(cfg)
    return SAC(
        "MultiInputPolicy",
        dummy_env_for_sac,
        learning_rate=cfg.training.learning_rate,
        buffer_size=cfg.training.buffer_size,
        learning_starts=cfg.training.learning_starts,
        batch_size=cfg.training.batch_size,
        tau=cfg.training.tau,
        train_freq=cfg.training.train_freq,
        gradient_steps=cfg.training.gradient_steps,
        gamma=cfg.training.gamma,
        verbose=0,
        device=device,
        policy_kwargs=policy_kwargs,
    )


def setup_run_outputs(cfg, model):
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir

    from stable_baselines3.common.logger import configure

    logger = configure(output_dir, ["csv"])
    model.set_logger(logger)
    snapshot_model_files(cfg.model_path, output_dir)
    writer = SummaryWriter(log_dir=output_dir)
    return output_dir, writer


def save_checkpoint(model, output_dir, step_name):
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save(os.path.join(checkpoint_dir, f"sac_manual_step_{step_name}"))


def broadcast_initial_weights(model, device, worker_queues):
    initial_weights = model.policy.to("cpu").state_dict()
    model.policy.to(device)
    for worker_queue in worker_queues:
        worker_queue.put(initial_weights)


def print_training_info(cfg, device):
    num_workers = cfg.training.num_workers
    target_total_steps = cfg.training.total_training_steps
    low_level_hz = cfg.control.frequencies.low_level
    training_seconds = target_total_steps / low_level_hz
    training_hours, remaining_seconds = divmod(training_seconds, 3600)
    training_minutes, training_seconds = divmod(remaining_seconds, 60)

    log.info(f"Initializing central learner with {num_workers} parallel workers on {device}...")
    print(
        f"Training for {target_total_steps} steps at {low_level_hz} Hz "
        f"equates to {int(training_hours)}h {int(training_minutes):02d}m {training_seconds:05.2f}s."
    )


def start_workers(
    cfg,
    output_dir,
    transition_queue,
    metrics_queue,
    worker_queues,
):
    workers = []
    for worker_id in range(cfg.training.num_workers):
        worker = mp.Process(
            target=worker_process,
            args=(
                worker_id,
                cfg,
                output_dir,
                transition_queue,
                metrics_queue,
                worker_queues[worker_id],
            ),
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)
    return workers


def worker_process(
    worker_id,
    cfg,
    output_dir,
    transition_queue,
    metrics_queue,
    weights_queue,
):
    """
    Subprocess isolated execution: Initializes env, runner, and an inference-only model.
    Steps physics and places transition chunks onto the queue.
    """
    ## 1. ----------------------
    torch.set_num_threads(1)

    log.info(f"Worker {worker_id}: Initializing Simulation...")
    env = make_env(cfg, output_dir)

    dummy_env_for_sac = DummySpaceEnv(cfg)
    policy_kwargs = build_sac_policy_kwargs(cfg)
    low_level_policy = SAC(
        "MultiInputPolicy",
        dummy_env_for_sac,
        buffer_size=1,
        device="cpu",
        policy_kwargs=policy_kwargs,
    )

    cartesian_policy = ScriptedCartesianPolicy(cfg)
    primitive_policy = ScriptedPrimitiveGeneratorPolicy(cfg)

    worker_transition_buffer = TransitionQueueBuffer(transition_queue)
    worker_metrics_buffer = MetricsQueueBuffer(metrics_queue)

    runner = EpisodeRunner(
        cfg=cfg,
        env=env,
        low_level_policy=low_level_policy,
        primitive_policy=primitive_policy,
        cartesian_policy=cartesian_policy,
        training=True,
        recorder=None,
        replay_buffer=worker_transition_buffer,
        metrics_queue=worker_metrics_buffer,
        weights_queue=weights_queue,
    )

    # Continuous episodes loop
    while True:
        # Sync weights before episode starts safely via the encapsulated runner policy
        runner._sync_weights()

        # Runner natively collects, chunks, calls policies, and populates `worker_transition_buffer` via `.add()`
        runner.run_episode(
            generate_primitives=True,
        )

        # Batch ship all collected physics steps to the central learner
        worker_transition_buffer.flush()
        worker_metrics_buffer.flush()


def _log_metrics(metrics_queue, writer, logging_step, recent_rewards):
    # Drain up to 10 reward batches per loop iteration to avoid starvation
    for _ in range(10):
        try:
            chunk_metrics = metrics_queue.get_nowait()
        except queue.Empty:
            break
        for metrics_dict in chunk_metrics:
            for key, val in metrics_dict.items():
                if isinstance(val, list):
                    # Write the mean if it's a detailed array, to avoid tensorboard clutter
                    if len(val) > 0:
                        writer.add_scalars(
                            f"rollout_detailed/{key}",
                            {
                                "mean": sum(val) / len(val),
                                "max": max(val),
                                "min": min(val),
                            },
                            logging_step,
                        )
                else:
                    if key == "total_reward":
                        recent_rewards.append(float(val))
                    writer.add_scalar(f"rollout/{key}", val, logging_step)
            logging_step += 1
    writer.flush()
    return logging_step


def _add_transition_and_train(chunk, model, sac_training_step, worker_queues, save_checkpoint):
    for t_obs, t_next_obs, t_action, t_reward, t_done in chunk:
        # empty dict is info field. done is passed as string to represent terminated
        model.replay_buffer.add(t_obs, t_next_obs, t_action, t_reward, t_done, [{}])
        sac_training_step += 1

        # 3. Perform learning identically to how SB3 behaves
        if sac_training_step > model.learning_starts and sac_training_step % model.train_freq.frequency == 0:
            model.train(gradient_steps=model.gradient_steps, batch_size=model.batch_size)

            if sac_training_step % 1000 == 0:
                model.logger.dump(step=sac_training_step)
                # Sync weights occasionally during heavy training (safely to CPU)
                cpu_state_dict = {k: v.cpu() for k, v in model.policy.state_dict().items()}
                for wq in worker_queues:  # TODO this whole thing seems pretty blocking
                    try:
                        # Clear old weights if the worker hasn't read them yet
                        while True:
                            wq.get_nowait()
                    except queue.Empty:
                        pass
                    wq.put(cpu_state_dict)

    # Intermediate saving logic equivalent
    if sac_training_step % 10000 == 0:
        save_checkpoint(f"step_{sac_training_step}")

    return sac_training_step


def _training_loop(
    cfg,
    metrics_queue,
    transition_queue,
    model,
    worker_queues,
    save_checkpoint,
    writer,
):
    target_total_steps = cfg.training.total_training_steps
    sac_training_step = 0
    logging_step = 0
    recent_rewards = deque(maxlen=100)
    progress = tqdm(
        total=target_total_steps,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
    )
    try:
        while sac_training_step < target_total_steps:
            logging_step = _log_metrics(metrics_queue, writer, logging_step, recent_rewards)

            # 1. Blocks until worker chunks arrive
            chunk = transition_queue.get()

            sac_training_step = _add_transition_and_train(chunk, model, sac_training_step, worker_queues, save_checkpoint)
            progress.update(sac_training_step - progress.n)
            if recent_rewards:
                progress.set_postfix(avg_reward=f"{sum(recent_rewards) / len(recent_rewards):.4f}")

    except KeyboardInterrupt:
        log.info("Keyboard interrupt, shutting down workers...")
    finally:
        progress.close()

    return sac_training_step


def run_distributed_training(cfg: DictConfig, device: torch.device):
    """
    Spawns worker processes to collect data using inference, while the main process
    updates a central target model and distributes updated weights.
    """
    print_training_info(cfg, device)

    # Set parallel method to spawn (required for torch/CUDA safety in multiprocessing)
    mp.set_start_method("spawn", force=True)

    transition_queue, metrics_queue, worker_queues = create_training_queues(cfg)
    model = create_central_sac_model(cfg, device)

    output_dir, writer = setup_run_outputs(cfg, model)

    broadcast_initial_weights(model, device, worker_queues)

    workers = start_workers(
        cfg,
        output_dir,
        transition_queue,
        metrics_queue,
        worker_queues,
    )

    sac_training_step = _training_loop(
        cfg,
        metrics_queue,
        transition_queue,
        model,
        worker_queues,
        lambda step_name: save_checkpoint(model, output_dir, step_name),
        writer,
    )

    save_checkpoint(model, output_dir, f"final_{sac_training_step}")

    writer.close()
    for w in workers:
        w.terminate()
