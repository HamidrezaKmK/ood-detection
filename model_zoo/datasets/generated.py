import numpy as np
from scipy.stats import vonmises
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle

import torch
import torch.nn.functional as f

from .supervised_dataset import SupervisedDataset
from model_zoo.utils import load_model_with_checkpoints
from model_zoo.density_estimator import DensityEstimator
from model_zoo import TwoStepDensityEstimator
from functools import lru_cache
from dotenv import load_dotenv

import typing as th
import os
from tqdm import tqdm
import shutil

import PIL

class DGMGeneratedDataset(SupervisedDataset):
    """Base class for generated datasets"""

    item_cache_size = 10000
    
    def __init__(
        self, 
        data_root: str,
        identifier: str,
        model_loading_config: th.Dict[str, th.Any],
        length: th.Optional[int] = None,
        seed: th.Optional[int] = None,
        generation_batch_size: int = 10,
        device: str = "cpu",
        model_root: th.Optional[str] = None,
    ):
        self.data_path = os.path.join(data_root, identifier)
        if model_root is None:
            load_dotenv()
            model_root = os.environ.get("MODEL_DIR", './runs/')
            
        def generate_all():
            # get the model and generate all the data and save it
            model: th.Union[DensityEstimator, TwoStepDensityEstimator] = \
                load_model_with_checkpoints(model_loading_config, device=device, root=model_root)
            
            if seed is not None:    
                np.random.seed(seed)
                torch.manual_seed(seed)
            rng = tqdm(range(0, length, generation_batch_size))
            rng.set_description("Generating data")
            for i in rng:
                r = min(i + generation_batch_size, length)
                data = model.sample(r - i).detach().cpu().permute(0, 2, 3, 1).numpy()
                for j, sample in enumerate(data):
                    img_dir = os.path.join(self.data_path, f"{i + j + 1}.png")
                    # save that sample in image format in img_dir
                    if sample.shape[-1] == 1:
                        im = PIL.Image.fromarray(sample[:, :, 0].astype(np.uint8), mode="L")
                    elif sample.shape[-1] == 3:
                        im = PIL.Image.fromarray(sample.astype(np.uint8))
                    else:
                        raise ValueError("Invalid number of channels: ", sample.shape[-1])
                    im.save(img_dir)
            self.length = length
                    
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            generate_all()
            
        else:
            # get the length of the dataset
            if length != len(os.listdir(self.data_path)):
                # delete all the content of the folder
                shutil.rmtree(self.data_path)
                os.mkdir(self.data_path)
                generate_all()
            else:
                print("Using existing data cached ...")
                self.length = len(os.listdir(self.data_path))
        
        self.cached_indices = {}
        
        self.device = device
    
    def get_data_min(self):
        return 0.0
    
    def get_data_max(self):
        return 255.0
    
    def get_data_shape(self):
        t = self.__getitem__(0)[0].shape
        return t
                   
    def __len__(self):
        return self.length

    def to(self, device):
        # uncache all the outputs
        if device != self.device:
            self.cached_indices = {}
            self.device = device

        return self
    
    def __getitem__(self, idx):
        if idx not in self.cached_indices:
            img_path = os.path.join(self.data_path, f"{idx + 1}.png")
            X = PIL.Image.open(img_path)
            # turn X into a numpy array of type float
            X = np.array(X).astype(np.float32)
            if len(X.shape) == 2:
                X = X[:, :, None]
            # turn X into a tensor
            X = torch.from_numpy(X).float().permute(2, 0, 1).to(self.device)
            # NOTE: if the dgm model is class conditional, then we can also generate the class labels here
            #       and return the sample conditioned on that specific class
            self.cached_indices[idx] = X, None, idx
        return self.cached_indices[idx]

def get_dgm_generated_datasets(data_root, dgm_args, identifier: th.Optional[str] = None):
    if identifier is None:
        checkpoint_dir = dgm_args['model_loading_config']['checkpoint_dir']
        # split checkpoint_dir according to the os path separator
        separated = checkpoint_dir.split('/')
        identifier = '_'.join(separated)
        # replace dots with underscores
        identifier = identifier.replace('.', '_')
        
    train_dset = DGMGeneratedDataset(data_root, identifier=identifier, **dgm_args)
    valid_dset = train_dset
    test_dset = train_dset
    return train_dset, valid_dset, test_dset
     
class Sphere(SupervisedDataset):
    """Sample from a Gaussian distribution projected to a sphere"""

    def __init__(self, name, role, manifold_dim=2, ambient_dim=3, size=1000, mu=None, sigma=None):
        assert role in ["train", "valid", "test"]

        self.name = name
        self.role = role
        
        self.manifold_dim = manifold_dim
        self.ambient_dim = ambient_dim

        if mu is None:
            mu = np.zeros(manifold_dim + 1)
            mu[0] = -1.0
            mu[1] = -1.0

        if sigma is None:
            sigma = np.diag(np.ones(manifold_dim + 1))
            
        gaussian_points = np.random.multivariate_normal(mu, sigma, size)
        sphere_points = gaussian_points / np.linalg.norm(gaussian_points, axis=1)[:,None]
        
        self.x = f.pad(torch.Tensor(sphere_points),
                            pad=(0, self.ambient_dim - self.manifold_dim - 1, 0, 0))
        self.y = manifold_dim * torch.ones(self.x.shape[0]).long()
    
    def get_data_min(self):
        return torch.min(self.x)

    def get_data_max(self):
        return torch.max(self.x)


class KleinBottle(SupervisedDataset):
    """Sample from a uniform distribution on a 2d Klein bottle immersed in 4d"""

    def __init__(self, name, role, theta_range=[0.0, 2*np.pi], phi_range=[0.0, 2*np.pi], size=1000, big_r=1, little_r=1):
        assert role in ["train", "valid", "test"]

        self.name = name
        self.role = role
        
        self.ambient_dim = 4
        self.theta_range = theta_range
        self.phi_range = phi_range
        self.big_r = big_r
        self.little_r = little_r

        self._generate_points(size)

    def _generate_points(self, size):
        klein_points = np.zeros((size, self.ambient_dim))
        # Rejection Sampling
        count = 0
        max_jacobian_determinant = self._jacobian_determinant(0)
        while count < size:
            theta = np.random.uniform(self.theta_range[0], self.theta_range[1])
            phi = np.random.uniform(self.phi_range[0], self.phi_range[1])
            eta = np.random.uniform(0.0, max_jacobian_determinant)
            if eta < self._jacobian_determinant(theta):  # else reject
                klein_points[count] = self._mobius_tube_transform(theta, phi)
                count += 1

        self.x = torch.Tensor(klein_points).float()
        self.y = torch.zeros(self.x.shape[0]).long()

    def _mobius_tube_transform(self, theta, phi):
        w = (self.big_r + self.little_r * np.cos(theta)) * np.cos(phi)
        x = (self.big_r + self.little_r * np.cos(theta)) * np.sin(phi)
        y = self.little_r * np.sin(theta) * np.cos(phi / 2)
        z = self.little_r * np.sin(theta) * np.sin(phi / 2)
        return [w, x, y, z]

    def _jacobian_determinant(self, theta):
        return self.little_r * np.sqrt((self.big_r + self.little_r*np.cos(theta))**2 + (0.5*self.little_r*np.sin(theta))**2)


def get_datasets_from_class(name, additional_instantiation_args: th.Optional[dict] = None):
    if name == "sphere":
        data_class = Sphere
    elif name == "klein":
        data_class = KleinBottle
    else:
        raise ValueError(f"Unknown dataset {name}")
    
    additional_instantiation_args = additional_instantiation_args or {}
    train_dset = data_class(name, "train", **additional_instantiation_args)
    valid_dset = data_class(name, "valid", **additional_instantiation_args)
    test_dset = data_class(name, "test", **additional_instantiation_args)
    
    return train_dset, valid_dset, test_dset


# Modified from https://github.com/jhjacobsen/invertible-resnet/blob/278faffe7bf25cd7488f8cd49bf5c90a1a82fc0c/models/toy_data.py#L8 
def get_2d_data(data, size):
    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=size, noise=1.0)[0]
        data = data[:, [0, 2]]
        data /= 5

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=size, factor=.5, noise=0.08)[0]
        data *= 3

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = size // 4
        n_samples1 = size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X)

        # Add noise
        data = X + np.random.normal(scale=0.08, size=X.shape)

    elif data == "8gaussians":
        dim = 2
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]
        for i in range(len(centers)):
          for k in range(dim-2):
            centers[i] = centers[i]+(0,)

        data = []
        for i in range(size):
            point = np.random.randn(dim) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            data.append(point)
        data = np.array(data)
        data /= 1.414

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = np.random.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        data = 2 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        data = x

    elif data == "checkerboard":
        x1 = np.random.rand(size) * 4 - 2
        x2_ = np.random.rand(size) - np.random.randint(0, 2, size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        data = np.concatenate([x1[:, None], x2[:, None]], 1)
        data *= 2

    elif data == "line":
        x = np.random.rand(size) * 5 - 2.5
        y = x
        data = np.stack((x, y), 1)
    elif data == "cos":
        x = np.random.rand(size) * 5 - 2.5
        y = np.sin(x) * 2.5
        data = np.stack((x, y), 1)

    elif data == "2uniforms":
        mixture_component = (np.random.rand(size) > 0.5).astype(int)
        x1 = np.random.rand(size) + mixture_component - 2*(1 - mixture_component)
        x2 = 2 * (np.random.rand(size) - 0.5)
        data = np.stack((x1, x2), 1)

    elif data == "2lines":
        x1 = np.empty(size)
        x1[:size//2] = -1.
        x1[size//2:] = 1.
        x1 += 0.01 * (np.random.rand(size) - .5)
        x2 = 2 * (np.random.rand(size) - 0.5)
        data = np.stack((x1, x2), 1)
        data = util_shuffle(data)

    elif data == "2marginals":
        x1 = np.empty(size)
        x1[:size//2] = -1.
        x1[size//2:] = 1.
        x1 += .5 * (np.random.rand(size) - .5)
        x2 = np.random.normal(size=size)
        data = np.stack((x1, x2), 1)
        data = util_shuffle(data)

    elif data == "1uniform":
        x1 = np.random.rand(size) - .5
        x2 = np.random.rand(size) - .5
        data = np.stack((x1, x2), 1)
        data = util_shuffle(data)

    elif data == "annulus":
        rad1 = 2
        rad2 = 1
        theta = 2 * np.pi * np.random.random(size)
        r = np.sqrt(np.random.random(size) * (rad1**2 - rad2**2) + rad2**2)
        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta)
        data = np.stack((x1, x2), 1)

    elif data == "sawtooth":
        u = np.random.rand(size)
        branch = u < .5
        x1 = np.zeros(size)
        x1[branch] = -1 - np.sqrt(1 - 2*u[branch])
        x1[~branch] = 1 + np.sqrt(2*u[~branch] - 1)
        x2 = np.random.rand(size)
        data = np.stack((x1, x2), 1)

    elif data == "quadspline":
        u = np.random.rand(size)
        branch = u < .5
        x1 = np.zeros(size)
        x1[branch] = -1 + np.cbrt(2*u[branch] - 1)
        x1[~branch] = 1 + np.cbrt(2*u[~branch] - 1)
        x2 = np.random.rand(size)
        data = np.stack((x1, x2), 1)

    elif data == "split-gaussian":
        x1 = np.random.normal(size=size)
        x2 = np.random.normal(size=size)
        x2[x1 >= 0] += 2
        x2[x1 < 0] -= 2
        data = np.stack((x1, x2), 1)

    elif data == "von-mises-circle":
        theta = vonmises.rvs(1, size=size, loc=np.pi/2)
        x1 = np.cos(theta)
        x2 = np.sin(theta)
        data = np.stack((x1, x2), 1)

    elif data == "sin-wave-mixture":
        theta_1 = 1.5*np.random.normal(size=size) - 3*np.pi/2
        theta_2 = 1.5*np.random.normal(size=size) + np.pi/2
        mixture_index = np.random.rand(size) < 0.5

        x1 = mixture_index*theta_1 + ~mixture_index*theta_2
        x2 = np.sin(x1)
        data = np.stack((x1, x2), 1)
        
    elif data == "two_moons":
        data = sklearn.datasets.make_moons(n_samples=size, noise=0.05)[0]

    else:
        assert False, f"Unknown dataset '{data}'"

    return torch.tensor(data, dtype=torch.get_default_dtype())


def get_simple_datasets(name, additional_instantiation_args: th.Optional[dict] = None):
    train_dset = SupervisedDataset(name, "train", get_2d_data(name, size=10000))
    valid_dset = SupervisedDataset(name, "valid", get_2d_data(name, size=1000))
    test_dset = SupervisedDataset(name, "test", get_2d_data(name, size=5000))
    return train_dset, valid_dset, test_dset


def get_generated_datasets(name, dataset_generation_args: th.Optional[dict] = None):    
    dataset_classes = ["sphere", "klein"]
    
    get_datasets_fn = get_datasets_from_class if name in dataset_classes else get_simple_datasets
    
    return get_datasets_fn(name, additional_instantiation_args=dataset_generation_args)
