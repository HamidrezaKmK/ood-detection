import torch
from typing import Any, Tuple
from .supervised_dataset import SupervisedDataset

class UnSupervisedDataset(torch.utils.data.Dataset):
    """A wrapper that omits the labels from a SupervisedDataset.
    Only used for the unsupervised methods for OOD detection as of now!"""

    def __init__(self, dset):
        self.dset = dset

    def __len__(self) -> int:
        return len(self.dset)

    def __getitem__(self, index: int):
        if isinstance(self.dset, SupervisedDataset):
            ret = self.dset[index][0]
            return ret
        else:
            return self.dset[index]
        
    def to(self, device):
        self.dset = self.dset.to(device)
        return self