import os
import torch
import multiprocessing as mp
import logging
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from stable_baselines3 import SAC

from robot_arm.envs.factory import make_env
from robot_arm.policies import WaypointPolicy
from robot_arm.coordinator import Coordinator

log = logging.getLogger(__name__)


def worker_process(worker_id, cfg, transition_queue, weights_dict_server, high_level_policy):
    """
    Subprocess isolated execution: Initializes env, coordinator, and an inference-only model.
    Steps physics and places transition chunks onto the queue.
    """
    ## 1. ----------------------
    torch.set_num_threads(1)

    log.info(f"Worker {worker_id}: Initializing Simulation...")
    # Workers do not need camera visual fidelity during low-level positional training
    env = make_env(cfg, minimize_visuals=True)

    # ???
    model = SAC("MultiInputPolicy", env, buffer_size=1, device="cpu")
    # what do i do here? pass the policy in?

    # high_level_policy = WaypointPolicy(
    #     trajectory_length=cfg.trajectory_length,
    #     speed=cfg.training.waypoint_speed,
    # )

    # this has to be combined with make env.
    coordinator = Coordinator(
        env=env,
        high_level_policy=high_level_policy,
        low_level_policy=model,
        high_level_hz=cfg.frequencies.high_level,
        low_level_hz=cfg.frequencies.low_level,
        training=True,
    )


    # Maybe we could just call coordinator or env.run?
    # env can get a recorder.
    # recorder should also record if the episode suddenly terminates.
    # recorder should probably just be a part of our env.

    # Continuous episodes loop
    # I have no idea how the whole thing with one thread works.
    # maybe I can place the env on one thread an still interact with it?
    while True:
        # 1. Sync weights before episode starts
        # training specific
        model.policy.load_state_dict(weights_dict_server["weights"])

        obs, info = env.reset()
        terminated = False
        truncated = False

        high_level_policy.generate_grab_waypoints(
            box_pose_6d=info["privileged_box_pose_6d"],
            lift_height=cfg.training.lift_height,
            gripper_open=cfg.training.gripper_open,
            gripper_closed=cfg.training.gripper_closed,
        )
        # TODO env.add_high level waypoints if cfg tells you to
        high_level_step = 0

        while not (terminated or truncated):
            # Coordinator natively chunks inside
            obs, reward, terminated, truncated, info = coordinator.step(
                obs, info, instruction="grab the box"
            )

            # Send the completed chunk to the learner thread
            transition_queue.put(info["low_level_transitions"])

            high_level_step += 1
            if high_level_step >= cfg.max_seconds * cfg.frequencies.high_level:
                break


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

    # 1. Main Learner Model Setup (We init one Env briefly just to map action bounds)
    dummy_env = make_env(cfg, minimize_visuals=True)

    # 2. Setup central SAC agent
    model = SAC(
        "MultiInputPolicy",
        dummy_env,
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
            target=worker_process, args=(i, cfg, transition_queue, weights_dict_server)
        )
        p.daemon = True
        p.start()
        workers.append(p)

    log.info("Starting central learning loop...")

    global_step = 0
    target_total_steps = cfg.training.total_training_steps

    try:
        while global_step < target_total_steps:
            # 1. Blocks until worker chunks arrive
            chunk = transition_queue.get()

            # 2. Add raw transitions to central buffer
            for t_obs, t_next_obs, t_action, t_reward, t_done, t_info in chunk:
                model.replay_buffer.add(
                    t_obs, t_next_obs, t_action, t_reward, t_done, [t_info]
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
