
import torch
import torch
from typing import Any, Tuple
from .supervised_dataset import SupervisedDataset
import requests
from tqdm import tqdm

SUPPORTED_IMAGE_DATASETS = [
    "celeba-small", 
    "celeba",
    "mnist", 
    "fashion-mnist", 
    "cifar10", 
    "cifar100",
    "svhn",
    "omniglot",
    "emnist-minus-mnist",
    "emnist",
    "tiny-imagenet",
]

SUPPORTED_GENERATED_DATASETS = [
    "sphere", "klein", "two_moons",
]

class TrainerReadyDataset(torch.utils.data.Dataset):
    # this is a dataset wrapper ready for being passed on to the trainer
    def __init__(self, dset):
        self.dset = dset
        
    def get_data_min(self):
        # check if self.dset has a method with the same name
        if hasattr(self.dset, "get_data_min"):
            return self.dset.get_data_min()
        if isinstance(self.dset, SupervisedDataset):
            return self.dset.x.min()
        raise Exception("No method get_data_min found in the dataset!")

    def get_data_max(self):
        if hasattr(self.dset, "get_data_max"):
            return self.dset.get_data_max()
        if isinstance(self.dset, SupervisedDataset):
            return self.dset.x.max()
        raise Exception("No method get_data_max found in the dataset!")
    
    def get_data_shape(self):
        if hasattr(self.dset, "get_data_shape"):
            return self.dset.get_data_shape()
        if isinstance(self.dset, SupervisedDataset):
            return self.dset.x.shape[1:]
        raise Exception("No method get_data_shape found in the dataset!")
    
    def tensorize_and_concatenate_all(self):
        if isinstance(self.dset, SupervisedDataset):
            return self.dset.x
        else:
            raise Exception("Only SupervisedDataset is supported for full tensorization now!")
    
    def to(self, device):
        self.dset = self.dset.to(device)
        return self 
    
    def __getitem__(self, index: int):
        ret = self.dset[index]
        if not isinstance(ret, tuple):
            return ret, -1, index
        else:
            return ret[0], ret[1], index
    
    def __len__(self) -> int:
        return len(self.dset)
    
    @property
    def device(self):
        if hasattr(self.dset, "device"):
            return self.dset.device
        if isinstance(self.dset, SupervisedDataset):
            return self.dset.x.device
        
class OmitLabels(torch.utils.data.Dataset):
    """A wrapper that omits the labels from a SupervisedDataset.
    Only used for the unsupervised methods for OOD detection as of now!"""

    def __init__(self, dset):
        self.dset = dset

    def __len__(self) -> int:
        return len(self.dset)

    def __getitem__(self, index: int):
        ret = self.dset[index]
        if isinstance(ret, tuple):
            return ret[0]
        else:
            return ret
        
    def to(self, device):
        self.dset = self.dset.to(device)
        return self
