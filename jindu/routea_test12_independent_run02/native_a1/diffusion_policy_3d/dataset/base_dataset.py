from typing import Dict

import torch
import torch.nn
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer

class BaseLowdimDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseLowdimDataset':
        # return an empty dataset by default
        return BaseLowdimDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: T, Do
            action: T, Da
        """
        raise NotImplementedError()


class BaseImageDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseImageDataset':
        # return an empty dataset by default
        return BaseImageDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key: T, *
            action: T, Da
        """
        raise NotImplementedError()

class BasePointcloudDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BasePointcloudDataset':
        # return an empty dataset by default
        return BasePointcloudDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key: T, *
            action: T, Da
        """
        raise NotImplementedError()


class BaseDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseDataset':
        # return an empty dataset by default
        return BaseDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key: T, *
            action: T, Da
        """
        raise NotImplementedError()


class BaseDroneDataset(BaseDataset):
    """无人机数据集基类"""
    def get_validation_dataset(self) -> 'BaseDroneDataset':
        return BaseDroneDataset()
    
    def __init__(self, 
                 n_obs_steps=2,
                 n_action_steps=8,
                 **kwargs):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        
    def get_drone_shape_meta(self):
        """返回无人机观测和动作形状元数据"""
        return {
            'obs': {
                'point_cloud': {'shape': [4096, 6], 'type': 'point_cloud'},
                'state': {'shape': [10], 'type': 'low_dim'},
                'goal_position': {'shape': [3], 'type': 'low_dim'}
            },
            'action': {'shape': [11], 'type': 'low_dim'}
        }