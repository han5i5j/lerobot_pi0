import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_constant_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from lerobot.common.policies.pi0.AccelerateFix import AsyncStep
from lerobot.common.policies.pi0.PreProcess import PreProcess
from lerobot.common.policies.pi0.RobomimicDataset import ComputeLimit
from lerobot.common.policies.pi0.DataPrefetcher import DataPrefetcher
from lerobot.common.policies.pi0.DiffusionPolicy import DiffusionPolicy
from lerobot.common.policies.pi0.FlorencePi0Net import FlorencePi0Net
# from mimictest.Simulation.ParallelMimic import ParallelMimic
from lerobot.common.policies.pi0.Train import train
from lerobot.common.policies.pi0.Train import evaluate
# from mimictest.Evaluation import Evaluation

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # Script-specific settings
    mode = 'train' # or 'eval'

    # Saving path
    save_path = Path('./Save/')
    save_path.mkdir(parents=True, exist_ok=True)

    # Dataset
    # abs_mode = True # relative EE action space or absolute EE action space
    # if abs_mode:
    #     file_name = 'image_abs.hdf5'
    # else:
    #     file_name = 'image.hdf5'
    # dataset_path = Path('/home/han5i5j/codes/mimictest/mimictest_data/robomimic_image/square/ph/') / file_name
    # bs_per_gpu = 192
    bs_per_gpu = 2
    # workers_per_gpu = 12
    workers_per_gpu = 2
    cache_ratio = 2

    repo_id = 'han5i5j1986/koch_lego_2024-10-16-01'
    delta_timestamps = {
        'observation.images.phone': [0, 1 / 30],
        'observation.images.laptop': [0, 1 / 30],
        'observation.state': [0, 1 / 30],
        'action': [0, 1 / 30],
    }

    # Space
    # limits = ComputeLimit(repo_id,delta_timestamps)
    num_actions_6d = 6
    lowdim_obs_dim = 6
    # lowdim_obs_dim = len(limits['low_dim_max'])
    obs_horizon = 2
    chunk_size = 2



    # delta_timestamps = {
    #     'observation.images.phone': [0, 1 / 30, 2 / 30, 3 / 30],
    #     'observation.images.laptop': [0, 1 / 30, 2 / 30, 3 / 30],
    #     'observation.state': [0, 1 / 30, 2 / 30, 3 / 30],
    #     'action': [0, 1 / 30, 2 / 30, 3 / 30],
    # }

    process_configs = {
        'observation.images.laptop': {
            'rgb_shape': (84, 84),
            'crop_shape': (76, 76),
            'max': torch.tensor(1.0),
            'min': torch.tensor(0.0),
        },
        'observation.images.phone': {
            'rgb_shape': (84, 84),
            'crop_shape': (76, 76),
            'max': torch.tensor(1.0),
            'min': torch.tensor(0.0),
        },
        'observation.state': {
            'max': torch.tensor(180),
            'min': torch.tensor(-180),
        },
        'action': {
            'max': torch.tensor(180),
            'min': torch.tensor(-180),
            # 'enable_6d_rot': True,
            # 'abs_mode': True,
        },
    }

    # Network
    model_path = Path("microsoft/Florence-2-base")
    freeze_vision_tower = True
    do_compile = False
    do_profile = False

    # Diffusion
    diffuser_train_steps = 10
    diffuser_infer_steps = 10
    diffuser_solver = "flow_euler"
    beta_schedule = None
    prediction_type = None
    clip_sample = None
    ema_interval = 10

    # Training
    # num_training_epochs = 1000
    num_training_epochs = 2
    # save_interval = 80
    save_interval = 1

    # load_epoch_id = 0
    load_epoch_id = 1

    gradient_accumulation_steps = 1
    lr_max = 1e-4
    warmup_steps = 5
    weight_decay = 1e-4
    max_grad_norm = 10
    # print_interval = 152
    print_interval = 1

    do_watch_parameters = False
    record_video = False
    loss_configs = {
        'action': {
            'loss_func': torch.nn.functional.mse_loss,
            'type': 'flow',
            'weight': 1.0,
            'shape': (chunk_size, num_actions_6d),
            # 'shape': (num_actions_6d,),
        },
    }

    # Testing (num_envs*num_eval_ep*num_GPU epochs)
    num_envs = 16
    # num_eval_ep = 6
    num_eval_ep = 2
    action_horizon = [0, 8]
    max_test_ep_len = 400

    # Preparation
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="wandb",
        kwargs_handlers=[kwargs],
        # cpu=True,
    )
    device = acc.device
    preprocessor = PreProcess(
        process_configs=process_configs,
        device=device,
    )
    # envs = ParallelMimic(dataset_path, num_envs, abs_mode)
    # eva = Evaluation(
    #     envs,
    #     num_envs,
    #     preprocessor,
    #     obs_horizon,
    #     action_horizon,
    #     save_path,
    #     device,
    # )
    # dataset = CustomMimicDataset(dataset_path, obs_horizon, chunk_size, start_ratio=0, end_ratio=1)


    # policy = act_koch_real
    # env = koch_real
    # hydra.run.dir = outputs / train / act_koch_lego_2025 - 01 - 0
    # 8
    # hydra.job.name = act_koch_lego
    # device = cuda
    # wandb.enable = true
    # resume = false

    dataset = LeRobotDataset(repo_id,delta_timestamps=delta_timestamps)

    loader = DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=bs_per_gpu,
        shuffle=True,
        num_workers=workers_per_gpu,
        drop_last=True,
    )

    net = FlorencePi0Net(
        path=model_path,
        freeze_vision_tower=freeze_vision_tower,
        num_actions=num_actions_6d,
        lowdim_obs_dim=lowdim_obs_dim,
    ).to(device)
    policy = DiffusionPolicy(
        net=net,
        loss_configs=loss_configs,
        do_compile=do_compile,
        scheduler_name=diffuser_solver,
        num_train_steps=diffuser_train_steps,
        num_infer_steps=diffuser_infer_steps,
        ema_interval=ema_interval,
        beta_schedule=beta_schedule,
        clip_sample=clip_sample,
        prediction_type=prediction_type,
    )
    policy.load_pretrained(acc, save_path, load_epoch_id)
    policy.load_wandb(acc, save_path, do_watch_parameters, save_interval)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr_max, weight_decay=weight_decay, fused=True)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    policy.net, policy.ema_net, optimizer, loader = acc.prepare(
        policy.net,
        policy.ema_net,
        optimizer,
        loader,
        device_placement=[True, True, True, False],
    )
    optimizer.step = AsyncStep
    prefetcher = DataPrefetcher(loader, device)

    if mode == 'train':
        train(
            acc=acc,
            prefetcher=prefetcher,
            preprocessor=preprocessor,
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            num_training_epochs=num_training_epochs,
            # eva=eva,
            num_eval_ep=num_eval_ep,
            max_test_ep_len=max_test_ep_len,
            device=device,
            save_path=save_path,
            load_epoch_id=load_epoch_id,
            save_interval=save_interval,
            print_interval=print_interval,
            bs_per_gpu=bs_per_gpu,
            max_grad_norm=max_grad_norm,
            record_video=record_video,
            do_profile=do_profile,
        )

        test_loader = DataLoader(
            dataset=dataset,
            sampler=None,
            batch_size=bs_per_gpu,
            shuffle=True,
            num_workers=workers_per_gpu,
            drop_last=True,
        )

        evaluate(acc,prefetcher,preprocessor,policy,device,num_eval_ep,bs_per_gpu,print_interval)





    # elif mode == 'eval':
    #     avg_reward = torch.tensor(eva.evaluate_on_env(
    #         acc=acc,
    #         policy=policy,
    #         epoch=0,
    #         num_eval_ep=num_eval_ep,
    #         max_test_ep_len=max_test_ep_len,
    #         record_video=True)
    #     ).to(device)
    #     avg_reward = acc.gather_for_metrics(avg_reward).mean(dim=0)
    #     acc.print(f'action horizon {action_horizon}, success rate {avg_reward}')
