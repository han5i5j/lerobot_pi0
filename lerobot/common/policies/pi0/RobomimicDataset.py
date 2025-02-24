import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


# from robomimic.utils.dataset import SequenceDataset
# from mimictest.Utils.PreProcess import action_euler_to_6d, action_axis_to_6d

def ComputeLimit(repo_id, delta_timestamps):
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

    # dataset = SequenceDataset(
    #     hdf5_path=dataset_path,
    #     obs_keys=(
    #         "robot0_eef_pos",
    #         "robot0_eef_quat",
    #         "robot0_gripper_qpos",
    #     ),
    #     dataset_keys=(
    #         "actions",
    #     ),
    #     load_next_obs=False,
    #     frame_stack=1,
    #     seq_length=1,
    #     pad_frame_stack=True,
    #     pad_seq_length=True,
    #     get_pad_mask=False,
    #     goal_mode=None,
    #     hdf5_cache_mode="all",
    #     hdf5_use_swmr=True,
    #     hdf5_normalize_obs=False,
    #     filter_by_attribute=None,
    # )
    states = []
    actions = []
    for i in range(len(dataset)):
        # state = np.concatenate((dataset[i]['obs']['robot0_eef_pos'], dataset[i]['obs']['robot0_eef_quat'], dataset[i]['obs']['robot0_gripper_qpos']), axis=-1)[0].astype(np.float32)
        state = dataset[i]['observation.state']
        states.append(state)
        # action = torch.from_numpy(dataset[i]['actions'][0])
        action = dataset[i]['action']
        # rot = action[3:6]
        # if abs_mode:
        #     action = torch.cat((action[:3], action_axis_to_6d(rot), action[6:]), dim=-1)
        # else:
        #     action = torch.cat((action[:3], action_euler_to_6d(rot), action[6:]), dim=-1)
        actions.append(action)
    states = torch.stack(states)
    actions = torch.stack(actions)
    return {
        "action_max": actions.max(dim=0)[0],
        "action_min": actions.min(dim=0)[0],
        "state_max": states.max(dim=0)[0],
        "state_min": states.min(dim=0)[0],
    }


# class CustomMimicDataset(Dataset):
#     def __init__(self, dataset_path, obs_horizon, chunk_size, start_ratio, end_ratio):
#         self.obs_dataset = SequenceDataset(
#             hdf5_path=dataset_path,
#             obs_keys=(
#                 "robot0_eef_pos",
#                 "robot0_eef_quat",
#                 "robot0_gripper_qpos",
#                 "object",
#                 "agentview_image",
#                 "robot0_eye_in_hand_image",
#             ),
#             dataset_keys=(
#                 "actions",
#             ),
#             load_next_obs=False,
#             frame_stack=obs_horizon,
#             seq_length=1,
#             pad_frame_stack=True,
#             pad_seq_length=True,
#             get_pad_mask=False,
#             goal_mode=None,
#             hdf5_cache_mode="all",
#             hdf5_use_swmr=True,
#             hdf5_normalize_obs=False,
#             filter_by_attribute=None,
#         )
#         self.action_dataset = SequenceDataset(
#             hdf5_path=dataset_path,
#             obs_keys=(
#             ),
#             dataset_keys=(
#                 "actions",
#             ),
#             load_next_obs=False,
#             frame_stack=1,
#             seq_length=chunk_size,
#             pad_frame_stack=True,
#             pad_seq_length=True,
#             get_pad_mask=False,
#             goal_mode=None,
#             hdf5_cache_mode="all",
#             hdf5_use_swmr=True,
#             hdf5_normalize_obs=False,
#             filter_by_attribute=None,
#         )
#         self.start_step = int(len(self.obs_dataset) * start_ratio)
#         self.end_step = int(len(self.obs_dataset) * end_ratio) - chunk_size
#
#     def __getitem__(self, idx):
#         idx = idx + self.start_step
#         batch = self.obs_dataset[idx]
#         action_batch = self.action_dataset[idx]
#         rgbs = torch.from_numpy(np.stack((batch['obs']['agentview_image'], batch['obs']['robot0_eye_in_hand_image']), axis=1))
#         low_dims = np.concatenate((batch['obs']['robot0_eef_pos'], batch['obs']['robot0_eef_quat'], batch['obs']['robot0_gripper_qpos']), axis=-1).astype(np.float32)
#         return {
#             'rgb': rearrange(rgbs, 't v h w c -> t v c h w').contiguous(),
#             'low_dim': low_dims,
#             'action': action_batch['actions'],
#         }
#
#     def __len__(self):
#         return self.end_step - self.start_step
