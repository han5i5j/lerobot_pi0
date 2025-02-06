from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn

from lerobot.common.policies.pi0.DiffusionPolicy import DiffusionPolicy
from lerobot.common.policies.pi0.FlorencePi0Net import FlorencePi0Net
from lerobot.common.policies.pi0.PreProcess import PreProcess
from lerobot.common.policies.pi0.configuration_florence_pi0 import FlorencePi0Config
from pathlib import Path
import torch

class FlorencePi0Policy(nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/han5i5j/lerobot_pi0",
    tags=["robotics", "pi0"],):

    name = 'florence_pi0'

    def __init__(self,
                 config: FlorencePi0Config | None = None,
                 dataset_stats: dict[str, dict[str, Tensor]] | None = None,):
        super().__init__()
        self.config = config
        self.init_preprocess()
        self.init_net()
        self.init_inner_policy()

    def init_preprocess(self):
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

        self.preprocessor = PreProcess(
            process_configs=process_configs,
            device=self.config.device,
        )

    def init_net(self):
        model_path = Path("microsoft/Florence-2-base")
        freeze_vision_tower = True
        num_actions_6d = 6
        lowdim_obs_dim = 6
        self.net = FlorencePi0Net(
            path=Path(model_path),
            freeze_vision_tower=freeze_vision_tower,
            num_actions=num_actions_6d,
            lowdim_obs_dim=lowdim_obs_dim,
        ).to(self.config.device)

    def init_inner_policy(self):
        # num_actions_6d = 6
        # do_compile = False
        loss_configs = {
            'action': {
                'loss_func': torch.nn.functional.mse_loss,
                'type': 'flow',
                'weight': 1.0,
                'shape': (self.config.chunk_size, self.config.num_actions_6d),
                # 'shape': (num_actions_6d,),
            },
        }
        # diffuser_train_steps = 10
        # diffuser_infer_steps = 10
        # diffuser_solver = "flow_euler"
        # beta_schedule = None
        # prediction_type = None
        # clip_sample = None
        # ema_interval = 10

        self.policy = DiffusionPolicy(
            net=self.net,
            loss_configs=loss_configs,
            do_compile=self.config.do_compile,
            scheduler_name=self.config.diffuser_solver,
            num_train_steps=self.config.diffuser_train_steps,
            num_infer_steps=self.config.diffuser_infer_steps,
            ema_interval=self.config.ema_interval,
            beta_schedule=self.config.beta_schedule,
            clip_sample=self.config.clip_sample,
            prediction_type=self.config.prediction_type,
        )


    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = self.preprocessor.process(batch, train=True)
        return self.policy.forward(batch)

    def reset(self):
        pass

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        batch = self.preprocessor.process(batch, train=False)
        pred = self.policy.forward(batch)
        real_pred = self.preprocessor.back_process(pred)
        return real_pred['action'][:,0,:]

    def update(self):
        if self.policy.use_ema:
            self.policy.update_ema()