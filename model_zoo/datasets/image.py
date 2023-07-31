import os
from pathlib import Path
from typing import Any, Tuple
import pandas as pd
import PIL
import torch
from torch.utils.data import Dataset
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
from model_zoo.datasets.supervised_dataset import SupervisedDataset
from functools import lru_cache
import numpy as np

class CelebA(Dataset):
    """
    CelebA PyTorch dataset
    The built-in PyTorch dataset for CelebA is outdated.
    """

    def __init__(self, root: str, role: str = "train", valid_fraction: float = 0.1, seed: int = 0):
        self.root = Path(root)
        self.role = role
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        celeb_path = lambda x: self.root / x

        role_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        splits_df = pd.read_csv(celeb_path("list_eval_partition.csv"))
        self.filename = splits_df[splits_df["partition"] == role_map[self.role]]["image_id"].tolist()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path = (self.root / "img_align_celeba" /
                    "img_align_celeba" / self.filename[index])
        X = PIL.Image.open(img_path)
        X = self.transform(X)

        return X, 0

    def __len__(self) -> int:
        return len(self.filename)
    
    def to(self, device):
        return self
    
class EMNISTMinusMNIST(Dataset):
    def __init__(
        self, 
        root: str, 
        role: str = "train", 
        valid_fraction: float = 0.1, 
        seed: int = 0
    ):
        
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"
        
        # Download and load the EMNIST dataset
        self.emnist_dataset = torchvision.datasets.EMNIST(
            root, 
            split='byclass', 
            train=True, 
            download=True,
        )

        # Get the indices of examples that are not digits (i.e., their labels are not between 0 and 9)
        self.indices = [i for i in range(len(self.emnist_dataset)) if self.emnist_dataset[i][1] >= 10]
        
        # shuffle indices with a fixed seed
        np.random.seed(seed)
        self.indices = np.random.permutation(np.array(self.indices))
        test_size = int(0.1 * len(self.indices))
        if role == 'test':
            self.indices = self.indices[:test_size]
        else:
            self.indices = self.indices[test_size:]
            valid_size = int(valid_fraction * len(self.indices))
            if role == 'train':
                self.indices = self.indices[valid_size:]
            else:
                self.indices = self.indices[:valid_size]
        
        self.transforms = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
            
    def get_data_min(self):
        return 0.0
    
    def get_data_max(self):
        return 255.0
    
    def get_data_shape(self):
        return (1, 28, 28)
          
    def __getitem__(self, index):
        # Get the image and label from the original EMNIST dataset for the given index
        image, label = self.emnist_dataset[self.indices[index]]
        image = 255.0 * self.transforms(image)
        # Subtract 10 from the label, so that labels start from 0 (not a necessary step, but often useful)
        return image.to(self.device), label - 10

    def __len__(self):
        return len(self.indices)
    
    def to(self, device):
        self.device = device
        return self
    
    
class Omniglot(Dataset):
    def __init__(self, root, role, valid_fraction, seed: int = 0):
        self.omniglot = torchvision.datasets.Omniglot(
            root=root, 
            background=True, 
            download=True
        )
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"
        # shuffle the dataset deterministically according to the splitting seed
        
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            perm = torch.randperm(len(self.omniglot))
            test_len = int(len(self.omniglot) * 0.1)
            valid_len = int(len(self.omniglot) * valid_fraction)
            test_indices = perm[:test_len]
            valid_indices = perm[test_len:valid_len+test_len]
            train_indices = perm[valid_len+test_len:]
            if role == "train":
                self.omniglot = torch.utils.data.Subset(self.omniglot, train_indices)
            elif role == 'valid':
                self.omniglot = torch.utils.data.Subset(self.omniglot, valid_indices)
            elif role == 'test':
                self.omniglot = torch.utils.data.Subset(self.omniglot, test_indices)
            else:
                raise ValueError(f"Unknown role {role}")
        
        self.cached_values = {}
            
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])  
        
    
    def get_data_min(self):
        return 0.0
    
    def get_data_max(self):
        return 255.0
    
    def get_data_shape(self):
        return (1, 28, 28)
           
    def __len__(self):
        return len(self.omniglot)

    def __getitem__(self, index) -> Any:
        if index not in self.cached_values:
            img, label = self.omniglot[index]
            img = self.transform(img)
            
            # turn label from an int to a tensor
            label = torch.tensor(label)
            
            # TODO: look into why I have to apply a x255 here!
            img = 255 * (1.0 - img).clamp(0.0, 1.0)
            
            self.cached_values[index] = img.to(self.device), label.to(self.device)
        return self.cached_values[index]
    
    def to(self, device):
        if device != self.device:
            self.cached_values = {}
        self.device = device
        return self
    
def get_image_datasets_by_class(dataset_name, data_root, valid_fraction, seed: int = 0):
    data_dir = os.path.join(data_root, dataset_name)

    data_class = {
        'celeba': CelebA,
        'omniglot': Omniglot,
        'emnist-minus-mnist': EMNISTMinusMNIST,
    }[dataset_name]

    
    train_dset = data_class(root=data_dir, role="train", valid_fraction=valid_fraction, seed=seed)
    valid_dset = data_class(root=data_dir, role="valid", valid_fraction=valid_fraction, seed=seed)
    test_dset = data_class(root=data_dir, role="test", valid_fraction=valid_fraction, seed=seed)

    return train_dset, valid_dset, test_dset


def image_tensors_to_dataset(dataset_name, dataset_role, images, labels):
    images = images.to(dtype=torch.get_default_dtype())
    labels = labels.long()
    return SupervisedDataset(dataset_name, dataset_role, x=images, y=labels)


# Returns tuple of form `(images, labels)`. Both are uint8 tensors.
# `images` has shape `(nimages, nchannels, nrows, ncols)`, and has
# entries in {0, ..., 255}
def get_raw_image_tensors(dataset_name, train, data_root):
    data_dir = os.path.join(data_root, dataset_name)

    if dataset_name == "LSUN":
        raise NotImplementedError("LSUN not implemented yet")
    
    elif dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True)
        images = torch.tensor(dataset.data).permute((0, 3, 1, 2))
        labels = torch.tensor(dataset.targets)

    elif dataset_name == "svhn":
        dataset = torchvision.datasets.SVHN(root=data_dir, split="train" if train else "test", download=True)
        images = torch.tensor(dataset.data)
        labels = torch.tensor(dataset.labels)

    elif dataset_name in ["mnist", "fashion-mnist"]:
        dataset_class = {
            "mnist": torchvision.datasets.MNIST,
            "fashion-mnist": torchvision.datasets.FashionMNIST,
        }[dataset_name]
        dataset = dataset_class(root=data_dir, train=train, download=True)
        images = dataset.data.unsqueeze(1)
        labels = dataset.targets
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return images.to(torch.uint8), labels.to(torch.uint8)


def get_torchvision_datasets(dataset_name, data_root, valid_fraction):
    images, labels = get_raw_image_tensors(dataset_name, train=True, data_root=data_root)

    perm = torch.randperm(images.shape[0])
    shuffled_images = images[perm]
    shuffled_labels = labels[perm]

    valid_size = int(valid_fraction * images.shape[0])
    valid_images = shuffled_images[:valid_size]
    valid_labels = shuffled_labels[:valid_size]
    train_images = shuffled_images[valid_size:]
    train_labels = shuffled_labels[valid_size:]

    train_dset = image_tensors_to_dataset(dataset_name, "train", train_images, train_labels)
    valid_dset = image_tensors_to_dataset(dataset_name, "valid", valid_images, valid_labels)
    
    test_images, test_labels = get_raw_image_tensors(dataset_name, train=False, data_root=data_root)
    test_dset = image_tensors_to_dataset(dataset_name, "test", test_images, test_labels)

    return train_dset, valid_dset, test_dset
    

def get_image_datasets(dataset_name, data_root, make_valid_dset):
    # Currently hardcoded; could make configurable
    valid_fraction = 0.1 if make_valid_dset else 0
    
    torchvision_datasets = ["mnist", "fashion-mnist", "svhn", "cifar10", "lsun"]
    
    get_datasets_fn = get_torchvision_datasets if dataset_name in torchvision_datasets else get_image_datasets_by_class
    
    return get_datasets_fn(dataset_name, data_root, valid_fraction)
