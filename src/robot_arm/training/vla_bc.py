import json
from pathlib import Path

from accelerate import Accelerator
import draccus
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import load_training_state, save_checkpoint, update_last_checkpoint

from robot_arm.cartesian_smolvla.configuration_cartesian_smolvla import CartesianSmolVLAConfig


def round_processors(policy_cfg, dataset_stats, initial: bool):
    if initial:
        return make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_cfg.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": policy_cfg.device},
                "normalizer_processor": {
                    "stats": dataset_stats,
                    "features": {**policy_cfg.input_features, **policy_cfg.output_features},
                    "norm_map": policy_cfg.normalization_mapping,
                },
            },
            postprocessor_overrides={
                "unnormalizer_processor": {
                    "stats": dataset_stats,
                    "features": policy_cfg.output_features,
                    "norm_map": policy_cfg.normalization_mapping,
                },
            },
        )
    # Saved processors define the same numerical representation across all rounds.
    return make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=policy_cfg.pretrained_path)


def batches_forever(loader):
    while True:
        yield from loader


def load_policy_config(cfg):
    if cfg.round_index == 0:
        policy_values = OmegaConf.to_container(cfg.bc.policy, resolve=True)
        policy_type = policy_values.pop("type")
        if policy_type != "cartesian_smolvla":
            raise ValueError(f"Expected cartesian_smolvla, received {policy_type}.")
        return draccus.decode(CartesianSmolVLAConfig, policy_values)
    model_dir = str(Path(cfg.previous_checkpoint) / "pretrained_model")
    policy_cfg = PreTrainedConfig.from_pretrained(model_dir)
    policy_cfg.pretrained_path = model_dir
    return policy_cfg


def build_training_config(cfg) -> TrainPipelineConfig:
    if cfg.bc.log_every_steps <= 0 or cfg.bc.batch_size <= 0 or cfg.bc.num_workers < 0:
        raise ValueError("Logging interval and batch size must be positive; worker count must be nonnegative.")
    if not 0 <= cfg.start_step < cfg.end_step <= cfg.total_steps:
        raise ValueError("Expected 0 <= start_step < end_step <= total_steps.")
    dataset_root = Path(cfg.dataset_root)
    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=dataset_root.name, root=str(dataset_root)),
        policy=load_policy_config(cfg),
        output_dir=Path(cfg.training_dir),
        job_name=cfg.bc.job_name,
        steps=cfg.total_steps,
        batch_size=cfg.bc.batch_size,
        num_workers=cfg.bc.num_workers,
        seed=cfg.seed,
    )
    train_cfg.validate()
    return train_cfg


class RoundTrainer:
    def __init__(self, cfg, train_cfg: TrainPipelineConfig):
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.accelerator = Accelerator(
            mixed_precision=cfg.bc.mixed_precision,
            cpu=train_cfg.policy.device == "cpu",
            step_scheduler_with_optimizer=False,
        )
        if self.accelerator.num_processes != 1:
            raise ValueError("DAgger rounds use one training process.")
        dataset = make_dataset(train_cfg)
        cartesian_policy = make_policy(train_cfg.policy, ds_meta=dataset.meta)
        self.preprocessor, self.postprocessor = round_processors(
            cartesian_policy.config, dataset.meta.stats, cfg.round_index == 0,
        )
        optimizer, self.scheduler = make_optimizer_and_scheduler(train_cfg, cartesian_policy)
        self.step = 0
        if cfg.round_index != 0:
            self.step, optimizer, self.scheduler = load_training_state(
                Path(cfg.previous_checkpoint), optimizer, self.scheduler,
            )
        if self.step != cfg.start_step:
            raise ValueError(f"Checkpoint step {self.step} does not match start_step {cfg.start_step}.")
        loader = DataLoader(
            dataset, batch_size=train_cfg.batch_size, num_workers=train_cfg.num_workers,
            shuffle=True, pin_memory=train_cfg.policy.device == "cuda", drop_last=False,
            # PyTorch requires zero timeout when loading in the main process.
            timeout=cfg.bc.dataloader_timeout_seconds if train_cfg.num_workers > 0 else 0,
        )
        self.cartesian_policy, self.optimizer, self.loader = self.accelerator.prepare(cartesian_policy, optimizer, loader)

    def update(self, batch):
        batch = self.preprocessor(batch)
        self.optimizer.zero_grad(set_to_none=True)
        with self.accelerator.autocast():
            loss, diagnostics = self.cartesian_policy.forward(batch)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("VLA training loss is non-finite.")
        self.accelerator.backward(loss)
        grad_norm = self.accelerator.clip_grad_norm_(
            self.cartesian_policy.parameters(), self.train_cfg.optimizer.grad_clip_norm,
        )
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        return {
            "step": self.step, **diagnostics, "grad_norm": float(grad_norm),
            "learning_rate": self.scheduler.get_last_lr()[0],
        }

    def train(self) -> None:
        self.cartesian_policy.train()
        self.train_cfg.output_dir.mkdir(parents=True, exist_ok=False)
        batches = batches_forever(self.loader)
        with (self.train_cfg.output_dir / "metrics.jsonl").open("w") as metrics, tqdm(
            total=self.cfg.end_step - self.step, desc=f"Train round {self.cfg.round_index}", unit="update",
        ) as progress:
            while self.step < self.cfg.end_step:
                diagnostics = self.update(next(batches))
                progress.update(1)
                if self.step % self.cfg.bc.log_every_steps == 0 or self.step == self.cfg.end_step:
                    metrics.write(json.dumps(diagnostics) + "\n")
                    metrics.flush()
                    progress.set_postfix(loss=diagnostics["loss"], completion_loss=diagnostics["completion_loss"])

    def save(self) -> None:
        checkpoint = self.train_cfg.output_dir / "checkpoints" / f"{self.step:08d}"
        save_checkpoint(
            checkpoint_dir=checkpoint, step=self.step, cfg=self.train_cfg,
            policy=self.accelerator.unwrap_model(self.cartesian_policy), optimizer=self.optimizer, scheduler=self.scheduler,
            preprocessor=self.preprocessor, postprocessor=self.postprocessor,
        )
        update_last_checkpoint(checkpoint)


def train_round(cfg) -> None:
    set_seed(cfg.seed)
    train_cfg = build_training_config(cfg)
    trainer = RoundTrainer(cfg, train_cfg)
    trainer.train()
    trainer.save()
    trainer.accelerator.end_training()
