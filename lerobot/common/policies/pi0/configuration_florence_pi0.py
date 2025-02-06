from dataclasses import dataclass, field


@dataclass
class FlorencePi0Config:

    device:str = 'cuda'
    num_actions_6d:int = 6
    lowdim_obs_dim:int = 6
    chunk_size:int = 2

    # Network
    model_path:str = "microsoft/Florence-2-base"
    freeze_vision_tower:bool = True
    do_compile:bool = False
    do_profile:bool = False

    # Diffusion
    diffuser_train_steps:int = 10
    diffuser_infer_steps:int = 10
    diffuser_solver:str = "flow_euler"
    beta_schedule:str = None
    prediction_type:str = None
    clip_sample:bool = None
    ema_interval:int = 10
