import os
import copy
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import EMAModel
import torch.nn as nn

class DiffusionPolicy(nn.Module):
    def __init__(
            self,
            net,
            loss_configs,
            do_compile,
            scheduler_name,
            num_train_steps,
            num_infer_steps,
            beta_schedule,
            clip_sample,
            prediction_type,
            ema_interval,
        ):
        super().__init__()

        self.do_compile = do_compile
        if do_compile:
            self.net = torch.compile(net)
        else:
            self.net = net
        self.loss_configs = loss_configs

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.net.parameters()))
        )

        self.use_ema = True
        self.ema_interval = ema_interval
        self.ema = EMAModel(
            parameters=self.net.parameters(),
            power=0.9999)
        self.ema_net = copy.deepcopy(self.net) 

        self.scheduler_name = scheduler_name
        self.prediction_type = prediction_type
        self.num_infer_steps = num_infer_steps
        if scheduler_name == 'ddpm':
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=num_train_steps,
                beta_schedule=beta_schedule,
                clip_sample=clip_sample,
                prediction_type=prediction_type,
            )
        elif scheduler_name == 'ddim':
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=num_train_steps,
                beta_schedule=beta_schedule,
                clip_sample=clip_sample,
                prediction_type=prediction_type,
            )
        elif scheduler_name == "flow_euler":
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=num_train_steps,
            )

    def compile(self, cache_size_limit=128):
        torch._dynamo.config.cache_size_limit = cache_size_limit
        torch._dynamo.config.optimize_ddp = False  # https://github.com/pytorch/pytorch/issues/104674
        # TODO: https://github.com/pytorch/pytorch/issues/109774#issuecomment-2046633776
        self.net = torch.compile(self.net)

    def _parameters(self):
        return filter(lambda p: p.requires_grad, self.net.parameters())

    def forward(self, batch):
        if self.training:
            return self._compute_loss(batch)
        else:
            return self._infer(batch)

    def _compute_loss(self, batch):
        device = batch['observation.state'].device
        B = batch['observation.state'].shape[0]

        # sample a diffusion iteration for each data point
        batch['timesteps'] = torch.randint(
            1, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device,
        ).long()

        noise = {}
        batch['noisy_inputs'] = {}
        for key in self.loss_configs:
            if self.loss_configs[key]['type'] == 'diffusion':
                noise[key] = torch.randn((B,)+self.loss_configs[key]['shape'], device=device)
                batch['noisy_inputs'][key] = self.noise_scheduler.add_noise(
                    sample=batch[key], 
                    noise=noise[key], 
                    timestep=batch['timesteps'],
                )
            elif self.loss_configs[key]['type'] == 'flow':
                noise[key] = torch.randn((B,)+self.loss_configs[key]['shape'], device=device)
                batch['noisy_inputs'][key] = self.noise_scheduler.scale_noise(
                    sample=batch[key], 
                    noise=noise[key], 
                    timestep=batch['timesteps'],
                )

        batch['obs_features'] = None

        pred, _ = self.net(batch)

        loss = {'total_loss': 0}
        for key in self.loss_configs:
            loss_func = self.loss_configs[key]['loss_func']
            weight = self.loss_configs[key]['weight']
            if self.loss_configs[key]['type'] == 'diffusion':
                if self.prediction_type == 'epsilon':
                    loss[key] = loss_func(pred[key], noise[key])
                elif self.prediction_type == 'sample':
                    loss[key] = loss_func(pred[key], batch[key])
                elif self.prediction_type == 'v_prediction':
                    target = self.noise_scheduler.get_velocity(batch[key], noise[key], batch['timesteps'])
                    loss[key] = loss_func(pred[key], target)
            elif self.loss_configs[key]['type'] == 'flow':
                loss[key] = loss_func(pred[key], noise[key] - batch[key])
            elif self.loss_configs[key]['type'] == 'simple':
                loss[key] = loss_func(pred[key], batch[key])
            loss['total_loss'] += loss[key] * weight
            loss['loss'] = loss['total_loss']
        return loss

    def _infer(self, batch):
        device = batch['observation.state'].device
        B = batch['observation.state'].shape[0]

        batch['noisy_inputs'] = {}
        for key in self.loss_configs:
            if self.loss_configs[key]['type'] == 'diffusion' or self.loss_configs[key]['type'] == 'flow':
                batch['noisy_inputs'][key] = torch.randn((B,)+self.loss_configs[key]['shape'], device=device)
        ones = torch.ones((B,), device=device).long()

        batch['obs_features'] = None
        self.noise_scheduler.set_timesteps(self.num_infer_steps)
        for k in self.noise_scheduler.timesteps:
            # A special fix because diffusers scheduler will add up _step_index, but we want it to kep the same at every k
            if "flow" in self.scheduler_name:
                current_step_index = self.noise_scheduler._step_index
            batch['timesteps'] = k*ones
            # if self.use_ema:
            #     out, batch['obs_features'] = self.ema_net(batch)
            # else:
            #     out, batch['obs_features'] = self.net(batch)

            out, batch['obs_features'] = self.net(batch)
            for key in self.loss_configs:
                if self.loss_configs[key]['type'] == 'diffusion' or self.loss_configs[key]['type'] == 'flow':
                    if "flow" in self.scheduler_name:
                        self.noise_scheduler._step_index = current_step_index
                    batch['noisy_inputs'][key] = self.noise_scheduler.step(
                        model_output=out[key],
                        timestep=k,
                        sample=batch['noisy_inputs'][key],
                    ).prev_sample

        pred = {}
        for key in out:
            if self.loss_configs[key]['type'] == 'diffusion' or self.loss_configs[key]['type'] == 'flow':
                pred[key] = batch['noisy_inputs'][key]
            elif self.loss_configs[key]['type'] == 'simple':
                pred[key] = out[key]
        return pred

    def update_ema(self):
        self.ema.step(self.net.parameters())
