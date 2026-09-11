import queue
import os
import multiprocessing as mp
import logging
from collections import deque
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

from robot_arm.envs.factory import make_env
from robot_arm.episode_runner import EpisodeRunner
from robot_arm.recording.git_snapshot import snapshot_git_state
from robot_arm.monitoring.gpu_monitor import GpuMonitor
from robot_arm.recording.model_snapshot import snapshot_model_files
from robot_arm.policies.numpy_policy import NumpySACPolicy
from robot_arm.policies.cartesian import ScriptedCartesianPolicy
from robot_arm.policies.primitive_generator import ScriptedPrimitiveGeneratorPolicy
from robot_arm.robot_schema import policy_observation_sizes
from robot_arm.monitoring.scalar_writer import ScalarWriter
from robot_arm.data.transition_loader import load_real_transitions

log = logging.getLogger(__name__)


# We use a custom local Replay queue to ship whole episodes, replacing the local SB3 replay buffer
class EpisodeQueueBuffer:
    def __init__(self, q):
        self.q = q
        self.episode = []

    def add(self, obs, next_obs, action, reward, done):
        self.episode.append((obs, next_obs, action, reward, done))

    def flush(self):
        if len(self.episode) > 0:
            self.q.put(self.episode)
            self.episode = []


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
    episode_queue = mp.Queue(maxsize=20)
    metrics_queue = mp.Queue(maxsize=1000)
    worker_queues = [mp.Queue(maxsize=1) for _ in range(cfg.training.num_workers)]
    return episode_queue, metrics_queue, worker_queues


def create_central_sac_model(cfg):
    from robot_arm.training.jax_sac import JaxSAC

    model = JaxSAC(cfg)
    if "continue_from" in cfg:
        model.load(cfg.continue_from)
    loaded = load_real_transitions(list(cfg.training.real_data.episode_paths), cfg, model.replay_buffer.real)
    if cfg.training.real_data.fraction > 0 and loaded == 0:
        raise ValueError("A positive real-data fraction requires real transitions at startup")
    log.info("Loaded %d real transitions; configured real sampling fraction: %.4f", loaded, cfg.training.real_data.fraction)
    return model


def setup_run_outputs(cfg):
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir

    snapshot_model_files(cfg.model_path, output_dir)
    snapshot_git_state(output_dir)
    writer = ScalarWriter(output_dir)
    return output_dir, writer


def save_checkpoint(model, output_dir, step_name):
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save(os.path.join(checkpoint_dir, f"jax_sac_{step_name}"))


def copy_policy_weights_to_cpu(model):
    return model.actor_params_numpy()


def broadcast_initial_weights(model, worker_queues):
    initial_weights = copy_policy_weights_to_cpu(model)
    for worker_queue in worker_queues:
        worker_queue.put(initial_weights)


def print_training_info(cfg):
    num_workers = cfg.training.num_workers
    target_total_steps = cfg.training.total_training_steps
    joint_hz = cfg.control.frequencies.joint
    training_seconds = target_total_steps / joint_hz
    training_hours, remaining_seconds = divmod(training_seconds, 3600)
    training_minutes, training_seconds = divmod(remaining_seconds, 60)

    log.info(f"Initializing central JAX learner with {num_workers} parallel workers...")
    print(f"Training for {target_total_steps} steps at {joint_hz} Hz " f"equates to {int(training_hours)}h {int(training_minutes):02d}m {training_seconds:05.2f}s.")


def start_workers(
    cfg,
    output_dir,
    episode_queue,
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
                episode_queue,
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
    episode_queue,
    metrics_queue,
    weights_queue,
):
    """
    Subprocess isolated execution: Initializes env, runner, and an inference-only model.
    Steps physics and places whole episodes onto the queue.
    """
    log.info(f"Worker {worker_id}: Initializing Simulation...")
    env = make_env(cfg, output_dir)
    initial_actor_params = weights_queue.get()
    observation_sizes = policy_observation_sizes(
        int(cfg.waypoint.cartesian_action_dim),
        env.policy_history_steps,
    )
    low_level_policy = NumpySACPolicy(initial_actor_params, observation_sizes, int(cfg.seed) + worker_id + 1)

    cartesian_policy = ScriptedCartesianPolicy(cfg)
    primitive_policy = ScriptedPrimitiveGeneratorPolicy(cfg)

    worker_episode_buffer = EpisodeQueueBuffer(episode_queue)
    worker_metrics_buffer = MetricsQueueBuffer(metrics_queue)

    runner = EpisodeRunner(
        cfg=cfg,
        env=env,
        low_level_policy=low_level_policy,
        primitive_policy=primitive_policy,
        cartesian_policy=cartesian_policy,
        training=True,
        recorder=None,
        replay_buffer=worker_episode_buffer,
        metrics_queue=worker_metrics_buffer,
        weights_queue=weights_queue,
        progress=None,
    )

    # Continuous episodes loop
    while True:
        # Sync weights before episode starts safely via the encapsulated runner policy
        runner._sync_weights()

        # Runner natively collects, calls policies, and populates `worker_episode_buffer` via `.add()`
        runner.run_episode(
            generate_primitives=True,
        )

        # Batch ship all collected physics steps to the central learner
        worker_episode_buffer.flush()
        worker_metrics_buffer.flush()


def _log_gpu(gpu_monitor, writer, sac_training_step):
    for key, value in gpu_monitor.latest().items():
        writer.add_scalar(f"gpu/{key}", value, sac_training_step)


def _log_metrics(metrics_queue, writer, sac_training_step, recent_rewards):
    # Drain up to 10 reward batches per loop iteration to avoid starvation
    for _ in range(10):
        try:
            chunk_metrics = metrics_queue.get_nowait()
        except queue.Empty:
            break
        for metrics_dict in chunk_metrics:
            for key, val in metrics_dict.items():
                if isinstance(val, list):
                    # add_scalars would write each series into its own run subdirectory, so the
                    # three are logged separately and left for tensorboard to group by tag.
                    if len(val) > 0:
                        writer.add_scalar(f"rollout_detailed/{key}/mean", sum(val) / len(val), sac_training_step)
                        writer.add_scalar(f"rollout_detailed/{key}/max", max(val), sac_training_step)
                        writer.add_scalar(f"rollout_detailed/{key}/min", min(val), sac_training_step)
                else:
                    if key == "total_reward":
                        recent_rewards.append(float(val))
                    writer.add_scalar(f"rollout/{key}", val, sac_training_step)
    writer.flush()


def _add_transition_and_train(episode, model, sac_training_step, worker_queues, writer):
    for t_obs, t_next_obs, t_action, t_reward, t_done in episode:
        model.replay_buffer.sim.add(t_obs, t_next_obs, t_action, t_reward, t_done)
        sac_training_step += 1

        if sac_training_step > model.learning_starts and sac_training_step % model.train_frequency == 0:
            learner_metrics = model.train(model.gradient_steps)
            for key, value in learner_metrics.items():
                writer.add_scalar(f"train/{key}", value, sac_training_step)

            if sac_training_step % model.broadcast_weights_every_n_steps == 0:
                cpu_state_dict = copy_policy_weights_to_cpu(model)
                for wq in worker_queues:  # TODO this whole thing seems pretty blocking
                    try:
                        # Clear old weights if the worker hasn't read them yet
                        while True:
                            wq.get_nowait()
                    except queue.Empty:
                        pass
                    wq.put(cpu_state_dict)

    return sac_training_step


def _log_replay_sizes(model, writer, step):
    writer.add_scalar("replay/sim_transitions", model.replay_buffer.sim.size, step)
    writer.add_scalar("replay/real_transitions", model.replay_buffer.real.size, step)


def _next_episode(episode_queue, workers, worker_check_seconds):
    """
    A worker only dies in simulation because of a bug, and the survivors hide it: they keep the
    queue full and the step counter climbing while the data rate silently drops. So the check runs
    every iteration rather than only when the queue runs dry, and the timeout exists to give it a
    turn rather than to detect anything.
    """
    while True:
        if not all(worker.is_alive() for worker in workers):
            raise RuntimeError("A collection worker exited; its traceback is above.")
        try:
            return episode_queue.get(timeout=worker_check_seconds)
        except queue.Empty:
            continue


def _training_loop(
    cfg,
    metrics_queue,
    episode_queue,
    model,
    worker_queues,
    workers,
    save_checkpoint,
    writer,
    gpu_monitor,
):
    target_total_steps = cfg.training.total_training_steps
    checkpoint_every_n_steps = cfg.training.checkpoint_every_n_steps
    sac_training_step = 0
    last_checkpoint_step = 0
    recent_rewards = deque(maxlen=100)
    progress = tqdm(
        total=target_total_steps,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
    )
    try:
        while sac_training_step < target_total_steps:
            _log_metrics(metrics_queue, writer, sac_training_step, recent_rewards)
            _log_gpu(gpu_monitor, writer, sac_training_step)

            # 1. Blocks until a worker episode arrives
            episode = _next_episode(episode_queue, workers, cfg.training.worker_check_seconds)

            sac_training_step = _add_transition_and_train(episode, model, sac_training_step, worker_queues, writer)
            _log_replay_sizes(model, writer, sac_training_step)
            # An episode arrives whole, so the step count steps over interval boundaries instead of
            # landing on them. Testing the distance since the last save is what makes this fire.
            if sac_training_step - last_checkpoint_step >= checkpoint_every_n_steps:
                save_checkpoint(f"step_{sac_training_step}")
                last_checkpoint_step = sac_training_step

            progress.update(sac_training_step - progress.n)
            if recent_rewards:
                progress.set_postfix(avg_reward=f"{sum(recent_rewards) / len(recent_rewards):.4f}")

    except KeyboardInterrupt:
        log.info("Keyboard interrupt, shutting down workers...")
    finally:
        progress.close()

    return sac_training_step


def shut_down(workers, queues, writer):
    """
    A queue keeps both ends of its pipe open in the process that created it, so a feeder thread
    part-way through writing a weight broadcast never sees the terminated worker close the read end.
    It blocks forever, and interpreter exit joins it. Cancelling that join is what lets the process
    die; closing the queue is not enough, because closing still waits for the buffer to flush.
    """
    for worker in workers:
        worker.terminate()
    for worker in workers:
        worker.join(timeout=5)
    for q in queues:
        q.cancel_join_thread()
        q.close()
    writer.close()


def run_distributed_training(cfg: DictConfig):
    """
    Spawns worker processes to collect data using inference, while the main process
    updates a central target model and distributes updated weights.
    """
    assert cfg.backend == "sim", "Online training workers must use simulation; real data is loaded from recordings"
    print_training_info(cfg)

    mp.set_start_method("spawn", force=True)

    episode_queue, metrics_queue, worker_queues = create_training_queues(cfg)
    model = create_central_sac_model(cfg)

    output_dir, writer = setup_run_outputs(cfg)
    _log_replay_sizes(model, writer, 0)

    broadcast_initial_weights(model, worker_queues)

    # os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"

    workers = start_workers(
        cfg,
        output_dir,
        episode_queue,
        metrics_queue,
        worker_queues,
    )

    gpu_monitor = GpuMonitor(cfg.training.gpu_sample_seconds)
    gpu_monitor.start()

    try:
        sac_training_step = _training_loop(
            cfg,
            metrics_queue,
            episode_queue,
            model,
            worker_queues,
            workers,
            lambda step_name: save_checkpoint(model, output_dir, step_name),
            writer,
            gpu_monitor,
        )

        save_checkpoint(model, output_dir, f"final_{sac_training_step}")
        print("Saved policy", flush=True)
    finally:
        gpu_monitor.stop()
        shut_down(workers, (episode_queue, metrics_queue, *worker_queues), writer)
