#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from threading import Lock

import hydra
import numpy as np
import torch
from deepdiff import DeepDiff
from omegaconf import DictConfig, ListConfig, OmegaConf
from termcolor import colored
from torch import nn
from torch.cuda.amp import GradScaler

from lerobot.common.datasets.factory import make_dataset, resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset
from lerobot.common.datasets.online_buffer import OnlineBuffer, compute_sampler_weights
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.logger import Logger, log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.policy_protocol import PolicyWithUpdate
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    set_global_seed,
)
from lerobot.scripts.eval import eval_policy
from transformers import get_constant_schedule_with_warmup
from typing import List
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    has_method,
    init_keyboard_listener,
    init_policy,
    log_control_info,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
from torch.nn.functional import mse_loss
import copy


def eval_policy(
    policy,
    batch,
    use_amp: bool = False,
):
    """Returns a dictionary of items for logging."""
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.eval()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        orgbatch = copy.deepcopy(batch)
        output_dict = policy.select_action(batch)
        loss = mse_loss(output_dict,orgbatch['action'][:,0,:])
        # loss = output_dict["loss"]

    info = {
        "loss": loss.item()
    }

    return info


def log_train_info(logger: Logger, info, step, cfg, dataset, is_online):
    loss = info["loss"]
    # grad_norm = info["grad_norm"]
    # lr = info["lr"]
    # update_s = info["update_s"]
    dataloading_s = info["dataloading_s"]

    # A sample is an (observation,action) pair, where observation and action
    # can be on multiple timestamps. In a batch, we have `batch_size`` number of samples.
    num_samples = (step + 1) * cfg.eval4real.batch_size
    avg_samples_per_ep = dataset.num_frames / dataset.num_episodes
    num_episodes = num_samples / avg_samples_per_ep
    num_epochs = num_samples / dataset.num_frames
    log_items = [
        f"step:{format_big_number(step)}",
        # number of samples seen during eval4real
        f"smpl:{format_big_number(num_samples)}",
        # number of episodes seen during eval4real
        f"ep:{format_big_number(num_episodes)}",
        # number of time all unique samples are seen
        f"epch:{num_epochs:.2f}",
        f"loss:{loss:.3f}",
        # f"grdn:{grad_norm:.3f}",
        # f"lr:{lr:0.1e}",
        # in seconds
        # f"updt_s:{update_s:.3f}",
        f"data_s:{dataloading_s:.3f}",  # if not ~0, you are bottlenecked by cpu or io
    ]
    logging.info(" ".join(log_items))

    info["step"] = step
    info["num_samples"] = num_samples
    info["num_episodes"] = num_episodes
    info["num_epochs"] = num_epochs
    info["is_online"] = is_online

    logger.log_dict(info, step, mode="train")


def eval(cfg: DictConfig, out_dir: str | None = None, job_name: str | None = None,policy_path: str | None = None,policy_overrides: List[str] | None = None,):
    init_logging()
    logging.info(pformat(OmegaConf.to_container(cfg)))

    # log metrics to terminal and wandb
    logger = Logger(cfg, out_dir, wandb_job_name=job_name)

    set_global_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_dataset")
    offline_dataset = make_dataset(cfg)
    if isinstance(offline_dataset, MultiLeRobotDataset):
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(offline_dataset.repo_id_to_index , indent=2)}"
        )

    logging.info("make_policy")

    policy, policy_fps, device, use_amp = init_policy(policy_path, policy_overrides)

    step = 0  # number of policy updates (forward + backward + optim)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    log_output_dir(out_dir)
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.eval4real.offline_steps=} ({format_big_number(cfg.eval4real.offline_steps)})")
    logging.info(f"{offline_dataset.num_frames=} ({format_big_number(offline_dataset.num_frames)})")
    logging.info(f"{offline_dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline eval4real
    if cfg.eval4real.get("drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            offline_dataset.episode_data_index,
            drop_n_last_frames=cfg.eval4real.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    dataloader = torch.utils.data.DataLoader(
        offline_dataset,
        num_workers=cfg.eval4real.num_workers,
        batch_size=cfg.eval4real.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.eval()
    offline_step = 0
    for _ in range(step, cfg.eval4real.offline_steps):
        if offline_step == 0:
            logging.info("Start offline eval4real on a fixed dataset")

        start_time = time.perf_counter()
        batch = next(dl_iter)
        dataloading_s = time.perf_counter() - start_time

        # for key in batch:
        #     batch[key] = batch[key].to(device, non_blocking=True)
        gpu_batch = {k: v.to(device) for k, v in batch.items()}
        batch = gpu_batch

        train_info = eval_policy(
            policy,
            batch,
        )
        train_info["dataloading_s"] = dataloading_s

        if step % cfg.eval4real.log_freq == 0:
            log_train_info(logger, train_info, step, cfg, offline_dataset, is_online=False)
        step += 1
        offline_step += 1  # noqa: SIM113
    if cfg.eval4real.offline_steps == 0:
        logging.info("End of eval4real")
        logger.finish()
        return


@hydra.main(version_base="1.2", config_name="default", config_path="../configs")
def eval_cli(cfg: dict):
    eval(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
        policy_path=hydra.core.hydra_config.HydraConfig.get().policypath
    )


# def train_notebook(out_dir=None, job_name=None, config_name="default", config_path="../configs"):
#     from hydra import compose, initialize
#
#     hydra.core.global_hydra.GlobalHydra.instance().clear()
#     initialize(config_path=config_path)
#     cfg = compose(config_name=config_name)
#     train(cfg, out_dir=out_dir, job_name=job_name)


if __name__ == "__main__":
    eval_cli()
