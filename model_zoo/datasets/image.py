import os
from pathlib import Path
from typing import Any, Tuple
import pandas as pd
import PIL
import torch
from torch.utils.data import Dataset
import torchvision.datasets
import torchvision.transforms as transforms
from model_zoo.datasets.supervised_dataset import SupervisedDataset
import numpy as np
import glob

class CelebATinyCropped(Dataset):
    """
    Dataset taken from 
    
    Kist, Andreas M. (2021, October 11). 
    CelebA Dataset cropped with Haar-Cascade face detector. 
    Zenodo. https://doi.org/10.5281/zenodo.5561092
    """
    def __init__(self, root: str, role: str = 'train', valid_fraction: float = 0.1, seed: int = 0):
        self.root = root 
        
        if not os.path.exists(root):
            print("Downloading CelebA dataset (clean and small version) ...")
            url = 'https://zenodo.org/record/5561092/files/celeba_5000_clean.zip?download=1'
            
            # Send HTTP request to the specified URL and save the response from server in a response object called r
            import requests
            r = requests.get(url)
            file_path = "celeba_5000_clean.zip"
            # Open the zip file in the write-binary mode
            with open(file_path, 'wb') as f:
                f.write(r.content)
                
            # now unzip
            from zipfile import ZipFile
            # Use ZipFile to extract
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)
            os.remove(file_path)
        
        self.image_paths = []
        # list all the files with .jpg extension
        self.all_image_files = glob.glob(os.path.join(self.root, 'celeba_5000_clean', '*.jpg'))
        
        np.random.seed(seed)
        randperm = np.random.permutation(len(self.all_image_files))
        train_len = int(0.9 * (1 - valid_fraction) * len(self.all_image_files))
        valid_len = int(0.9 * valid_fraction * len(self.all_image_files))
        if role == 'train':
            self.all_image_files = [self.all_image_files[i] for i in randperm[:train_len]]
        elif role == 'valid':
            self.all_image_files = [self.all_image_files[i] for i in randperm[train_len:train_len + valid_len]]
        elif role == 'test':
            self.all_image_files = [self.all_image_files[i] for i in randperm[train_len + valid_len:]]
        else:
            raise ValueError(f"Unknown role {role}")
        
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        
        self.cached_values = {}
        
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"

    def __len__(self):
        return len(self.all_image_files)

    def to(self, device):
        if device != self.device:
            self.cached_values = {}
        self.device = device
        return self
    
    def get_data_min(self):
        return 0.0
    
    def get_data_max(self):
        return 255.0
    
    def get_data_shape(self):
        return (3, 32, 32)
    
    def __getitem__(self, idx):
        if idx not in self.cached_values:
            
            img_name = self.all_image_files[idx]
            image = PIL.Image.open(img_name).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            self.cached_values[idx] = torch.clamp(image.to(self.device) * 255, min=0, max=255)
            
        return self.cached_values[idx]
    
class CelebA(Dataset):
    """
    CelebA PyTorch dataset
    The built-in PyTorch dataset for CelebA is outdated.
    """

    def __init__(self, root: str, role: str = "train", valid_fraction: float = 0.1, seed: int = 0):
        self.celeba_dataset = torchvision.datasets.CelebA(
            root=root,
            split=role,
            download=True,
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
        )
                
        self.root = Path(root)
        self.role = role
        
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.celeba_dataset[index][0].to(self.device), self.celeba_dataset[index][1].to(self.device)

    def __len__(self) -> int:
        return len(self.celeba_dataset)
    
    def get_data_min(self):
        return 0.0
    
    def get_data_max(self):
        return 255.0
    
    def get_data_shape(self):
        return (3, 32, 32)
    
    def to(self, device):
        self.device = device
        return self
    
class EMNIST(Dataset):
    def __init__(
        self, 
        root: str, 
        role: str = "train", 
        valid_fraction: float = 0.1, 
        seed: int = 0,
        classes_to_ignore: int = 0, 
    ):
        
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"
        self.classes_to_ignore = classes_to_ignore
        # Download and load the EMNIST dataset
        self.emnist_dataset = torchvision.datasets.EMNIST(
            root, 
            split='byclass', 
            train=True, 
            download=True,
        )

        # Get the indices of examples that are not digits (i.e., their labels are not between 0 and 9)
        self.indices = [i for i in range(len(self.emnist_dataset)) if self.emnist_dataset[i][1] >= self.classes_to_ignore]
        
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
        return image.to(self.device), label - self.classes_to_ignore

    def __len__(self):
        return len(self.indices)
    
    def to(self, device):
        self.device = device
        return self

class EMNISTMinusMNIST(EMNIST):
    def __init__(
        self, 
        root: str, 
        role: str = "train", 
        valid_fraction: float = 0.1, 
        seed: int = 0,
    ):
        super().__init__(
            root=root,
            role=role,
            valid_fraction=valid_fraction,
            seed=seed,
            classes_to_ignore=10,
        )
    
    
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

# NOTE: this is incomplete
class LSUN(Dataset):
    def __init__(self, root, role, valid_fraction, seed: int = 0):
        self.omniglot = torchvision.datasets.LSUN(
            root=root, 
            classes={'train':'train', 'valid':'val', 'test':'test'}[role],
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]),
            download=True,
        )
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"
        # shuffle the dataset deterministically according to the splitting seed
        
        self.cached_values = {}
        
    
    def get_data_min(self):
        return 0.0
    
    def get_data_max(self):
        return 255.0
    
    def get_data_shape(self):
        return (3, 32, 32)
           
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
    
class TinyImageNet(Dataset):
    def __init__(self, root, role, valid_fraction, seed: int = 0):
        #self, root_dir, n_images, train_or_test, transform=None, ):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.root_dir = os.path.join(root, 'tiny-imagenet')
        if not os.path.exists(self.root_dir):
            # download the dataset from http://cs231n.stanford.edu/tiny-imagenet-200.zip
            # create the root directory and extract the zip file in it
            # then erase the zip file
            
            os.makedirs(self.root_dir)
            # now download
            # URL of the dataset
            print("Downloading dataset ...")
            
            url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

            # Send HTTP request to the specified URL and save the response from server in a response object called r
            import requests
            r = requests.get(url)
            file_path = "tiny-imagenet-200.zip"
            # Open the zip file in the write-binary mode
            with open(file_path, 'wb') as f:
                f.write(r.content)
                
            # now unzip
            from zipfile import ZipFile
            # Use ZipFile to extract
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
            os.remove(file_path)
            
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        if role in ['train', 'valid']:
            folds = ('train', 'val',)
        else:
            folds = ('test',)

        all_image_files = []
        for fold in folds:
            if fold == 'train':
                # /h/hamidk/ood-detection/data/tiny-imagenet/tiny-imagenet/tiny-imagenet-200
                # print("root", os.path.join(self.root_dir, "tiny-imagenet-200", fold, '*'))
                subfolders = glob.glob(os.path.join(self.root_dir, "tiny-imagenet-200", fold, '*'))
                # print("Subfolders", subfolders)
                for f in subfolders:
                    image_files = glob.glob(os.path.join(f, 'images/', '*.JPEG'))
                    # print(len(image_files))
                    all_image_files.extend(image_files)
            else:
                image_files = glob.glob(os.path.join(self.root_dir, "tiny-imagenet-200", fold, 'images', '*.JPEG'))
                
                all_image_files.extend(image_files)
        self.all_image_files = all_image_files
        
        np.random.seed(seed)
        randperm = np.random.permutation(len(self.all_image_files))
        train_split = int((1 - valid_fraction) * len(self.all_image_files))
        if role == 'train':
            self.all_image_files = [self.all_image_files[i] for i in randperm[:train_split]]
        elif role == 'valid':
            self.all_image_files = [self.all_image_files[i] for i in randperm[train_split:]]
            
        self.cached_values = {}
        
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"

    def __len__(self):
        return len(self.all_image_files)

    def to(self, device):
        if device != self.device:
            self.cached_values = {}
        self.device = device
        return self
    
    def get_data_min(self):
        return 0.0
    
    def get_data_max(self):
        return 255.0
    
    def get_data_shape(self):
        return (3, 32, 32)
    
    def __getitem__(self, idx):
        if idx not in self.cached_values:
            
            img_name = self.all_image_files[idx]
            image = PIL.Image.open(img_name).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            self.cached_values[idx] = torch.clamp(image.to(self.device) * 255, min=0, max=255)
            
        return self.cached_values[idx]

def get_image_datasets_by_class(dataset_name, data_root, valid_fraction, seed: int = 0):
    data_dir = os.path.join(data_root, dataset_name)

    data_class = {
        'celeba': CelebA,
        'celeba-small': CelebATinyCropped,
        'omniglot': Omniglot,
        'emnist-minus-mnist': EMNISTMinusMNIST,
        'emnist': EMNIST,
        "tiny-imagenet": TinyImageNet,
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
    
    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True)
        images = torch.tensor(dataset.data).permute((0, 3, 1, 2))
        labels = torch.tensor(dataset.targets)
        
    elif dataset_name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(root=data_dir, train=train, download=True)
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
    
    torchvision_datasets = ["mnist", "fashion-mnist", "svhn", "cifar10", "cifar100"]
    
    get_datasets_fn = get_torchvision_datasets if dataset_name in torchvision_datasets else get_image_datasets_by_class
    
    return get_datasets_fn(dataset_name, data_root, valid_fraction)
