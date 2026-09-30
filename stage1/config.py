"""Hydra/LazyConfig entry for Stage-1 training with mimic-video's trainer.

  source stage1/env.sh; cd $MIMIC_MODEL
  torchrun --nproc_per_node=2 -m scripts.train --config=stage1_config.py -- experiment=stage1_clean \
      [model.config.hand_weight=100] [dataloader_train.batch_size=2] [trainer.grad_accum_iter=8] [job.name=...]

Registers on top of mimic-video's make_config():
  model               stage1_model          Stage1Model (pretrained Cosmos 2B 480p 10fps, DDP)
  video_dataset_train stage1 / video_dataset_val stage1   Stage1Dataset on $UNIDEX_REPO/stage1_data
  dataloader_train    stage1                dataset-balanced WeightedResumableSampler
  dataloader_val      stage1
  experiment          stage1_clean | stage1_leaky | stage1_off   (hand_mode; everything else identical)
Defaults follow mimic-video's video finetuning (LoRA rank 256, alpha 32, their target modules, lr 1.778e-4,
fusedadamw, constant schedule, high_sigma_ratio 0.05, loss_scale 100).
Environment: UNIDEX_WANDB_MODE (default disabled), UNIDEX_WANDB_ENTITY, UNIDEX_STAGE1_DATA (default
$UNIDEX_REPO/stage1_data), UNIDEX_T5_SUBDIR (default t5; t5_TESTONLY only for local smoke tests).
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader

from cosmos_predict2.callbacks.video_eval import VideoEvalCallback
from cosmos_predict2.configs.config import make_config as _mimic_make_config
from cosmos_predict2.configs.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.models.video2world_model import Predict2ModelManagerConfig
from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
from imaginaire.lazy_config import PLACEHOLDER
from imaginaire.lazy_config import LazyCall as L

from stage1.dataset import Stage1Dataset
from stage1.metrics_callback import Stage1Metrics
from stage1.model import Stage1Model, Stage1ModelConfig
from stage1.sampler import get_weighted_sampler

REPO = Path(os.environ.get("UNIDEX_REPO", Path(__file__).resolve().parents[1]))
DATA = Path(os.environ.get("UNIDEX_STAGE1_DATA", REPO / "stage1_data"))
T5_SUBDIR = os.environ.get("UNIDEX_T5_SUBDIR", "t5")
LORA_TARGETS = "q_proj,k_proj,v_proj,output_proj,x_embedder.proj.1,linear_1,linear_2,mlp.layer1,mlp.layer2"


def get_local_batch_size(global_bsz: int) -> int:
    res = global_bsz / parallel_state.get_data_parallel_world_size()
    if not res.is_integer():
        raise ValueError(f"global batch {global_bsz} not divisible by the number of GPUs")
    return int(res)


def _pipe_config():
    cfg = get_cosmos_predict2_video2world_pipeline(model_size="2B", resolution="480", fps=10)
    cfg.guardrail_config.enabled = False
    cfg.ema.enabled = False
    return cfg


STAGE1_MODEL = dict(
    trainer=dict(distributed_parallelism="ddp"),
    model=L(Stage1Model)(
        config=Stage1ModelConfig(
            pipe_config=_pipe_config(),
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path=get_cosmos_predict2_video2world_checkpoint(model_size="2B", resolution="480", fps=10),
                text_encoder_path=""),
            train_architecture="lora", lora_rank=256, lora_alpha=32, init_lora_weights=True,
            lora_target_modules=LORA_TARGETS, fsdp_shard_size=0, high_sigma_ratio=0.05, loss_scale=100.0),
        _recursive_=False),
)

train_ds = L(Stage1Dataset)(root=str(DATA), repo=str(REPO), is_val=False, t5_subdir=T5_SUBDIR)
val_ds = L(Stage1Dataset)(root=str(DATA), repo=str(REPO), is_val=True, t5_subdir=T5_SUBDIR, seed=0)
dataloader_train = L(DataLoader)(dataset="${video_dataset_train}",
                                 sampler=L(get_weighted_sampler)(dataset="${video_dataset_train}"),
                                 batch_size=2, drop_last=True, num_workers=8, prefetch_factor=4, pin_memory=True,
                                 persistent_workers=True)
dataloader_val = L(DataLoader)(dataset="${video_dataset_val}", batch_size=2, shuffle=False, drop_last=False,
                               num_workers=4, pin_memory=False, persistent_workers=False)

BASE = dict(
    defaults=[
        {"override /model": "stage1_model"},
        {"override /video_dataset_train": "stage1"},
        {"override /video_dataset_val": "stage1"},
        {"override /dataloader_train": "stage1"},
        {"override /dataloader_val": "stage1"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "constant"},
        {"override /ckpt_type": "standard"},
        "_self_",
    ],
    job=dict(project="unidex_stage1", group="stage1", name=""),
    model=dict(config=dict(hand_mode="clean")),
    model_parallel=dict(cpu_offloading_activations=False, cpu_offloading_weights=False),
    optimizer=dict(lr=1.778e-4),
    checkpoint=dict(save_iter=1000),
    trainer=dict(
        distributed_parallelism="ddp", grad_accum_iter=4, max_iter=30_000, logging_iter=50,
        validation_iter=1000, max_val_iter=50, run_validation=True,
        epoch_checkpoint_throttling_min_period_minutes=30,
        callbacks=dict(
            video_eval=L(VideoEvalCallback)(fuse_lora=True),
            stage1_metrics=L(Stage1Metrics)(config=PLACEHOLDER, trainer=PLACEHOLDER, every_n="${trainer.logging_iter}"),
            wandb=dict(mode=os.environ.get("UNIDEX_WANDB_MODE", "disabled"),
                       entity_name=os.environ.get("UNIDEX_WANDB_ENTITY", "none"), project_name="unidex_stage1"),
        ),
    ),
)


def register_stage1() -> None:
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="stage1_model", node=STAGE1_MODEL)
    cs.store(group="video_dataset_train", package="video_dataset_train", name="stage1", node=train_ds)
    cs.store(group="video_dataset_val", package="video_dataset_val", name="stage1", node=val_ds)
    cs.store(group="dataloader_train", package="dataloader_train", name="stage1", node=dataloader_train)
    cs.store(group="dataloader_val", package="dataloader_val", name="stage1", node=dataloader_val)
    for mode in ("clean", "leaky", "off"):
        cfg = copy.deepcopy(BASE)
        cfg["model"]["config"]["hand_mode"] = mode
        cfg["job"]["name"] = f"stage1_{mode}"
        cs.store(group="experiment", package="_global_", name=f"stage1_{mode}", node=cfg)


def make_config():
    c = _mimic_make_config()
    register_stage1()
    return c
