"""
This file contains classes that do statistical analysis in the latent space of a model.
They might also be used for other purposes such as sampling, calculating the cdf of a specific
volume in the latent space, etc.
"""

import torch
import typing as th
import dypy as dy
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from scipy.special import logsumexp
from .utils import stack_back_iterables

class LatentStatsCalculator:
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        # Encoding and decoding model
        encoding_model_class: th.Optional[str] = None, 
        encoding_model_args: th.Optional[th.Dict[str, th.Any]] = None,
        # verbosity
        verbose: int = 0,
    ):
        self.likelihood_model = likelihood_model
        self.verbose = verbose
        self.encoding_model = dy.eval(encoding_model_class)(likelihood_model, **(encoding_model_args or {}))
    
    def calculate_statistics(self, r, loader, use_cache: bool = False) -> np.ndarray:
        """
        This function calculates some sort of latent statistics w.r.t a radius 'r'
        """
        raise NotImplementedError("You must implement a calculate_statistics function")
    
def calculate_ellipsoids(
    stats_calculator: LatentStatsCalculator, 
    loader, 
    return_rotations: bool = False, 
    stack_back: bool = False, 
    device = None
):
    """
    This function takes in a loader of the form
    [
        (x_batch_1),
        (x_batch_2),
        ...,
        (x_batch_n)
    ]
    where each x_batch_i is a batch of data points.
    
    This function calculates the ellipsoids for each of the data points in the batch.
    And optimizes some of the computations by doing either vmap or batching if specified.
    After that, it returns the centers and radii of the ellipsoids in the following format.
    
    centers = [centers_batch_1, centers_batch_2, ..., centers_batch_n]
    radii = [radii_batch_1, radii_batch_2, ..., radii_batch_n]
    
    Where each of the centers_batch_i and radii_batch_i 
    have the shape (batch_size, latent_dim)
    """
    centers = []
    radii = []
    if return_rotations:
        rotations = []
    
    jax, z_values = stats_calculator.encoding_model.calculate_jacobian(loader, stack_back=False)
    
    if stats_calculator.verbose > 1:
        iterable = tqdm(zip(jax, z_values))
        iterable.set_description("Calculating the ellipsoids")
    else:
        iterable = zip(jax, z_values)
    
    device = device or stats_calculator.likelihood_model.device
    
    for jac, z in iterable: 
        # turn the devices
        jac = jac.to(device)
        z = z.to(device)
        
        ellipsoids = torch.matmul(jac.transpose(1, 2), jac)
        
        L, Q = torch.linalg.eigh(ellipsoids)
        # Each row of L would contain the scales of the non-central chi-squared distribution
        center = torch.matmul(Q.transpose(1, 2), z.unsqueeze(-1)).squeeze(-1)
        
        if return_rotations:
            rotations.append(Q.transpose(1, 2).cpu().detach())
            
        centers.append(center.cpu().detach())
        radii.append(L.cpu().detach())
    
    # return the rotations that cauesed the centers to be what they are too
    all_ret = [centers, radii]
    if return_rotations:
        return all_ret + [rotations]
    
    return stack_back_iterables(loader, *all_ret) if stack_back else all_ret
    
class ParallelogramStatsCalculator(LatentStatsCalculator):
    def __init__(
        self,
        *args,
        filtering: th.Optional[th.Union[th.Dict[str, str], str]] = None,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        # used for log correction
        self.eps = eps
        
        if filtering is None:
            self.filtering = lambda cent, rad: torch.zeros_like(rad, dtype=torch.bool)
        elif isinstance(filtering, str):
            self.filtering = dy.eval(filtering)
        elif isinstance(filtering, dict):
            self.filtering = dy.eval_function(**filtering)
        else:
            raise ValueError("filtering should be either a string or a dictionary.")
    
    
    def calculate_single_cdf(self, r, loader, use_cache: bool = False) -> np.ndarray:
        # You should implement your own function here
        return np.exp(self.calculate_single_log_cdf(r, loader, use_cache))
    
    def calculate_single_log_cdf(self, r, loader, use_cache: bool = False) -> float:
        if not use_cache:
            self.centers, self.radii = calculate_ellipsoids(self, loader)
        if not hasattr(self, 'centers') or not hasattr(self, 'radii'):
            raise ValueError("You should first calculate the ellipsoids before calculating the CDFs with cache.")
        centers, radii = self.centers, self.radii
        def _check_r_primitive():
            if isinstance(r, float):
                return True
            if isinstance(r, int):
                return True
        if self.verbose > 1:
            print("Calculating the single CDFs...")
            if _check_r_primitive():
                print(f"\tr = {r}")
            
        if not use_cache or not hasattr(self, 'rects'):
            # if it is not cached, then we should calculate the rects
            self.rects = []
            for c_batch, rad_batch in zip(centers, radii):
                for c, rad in zip(c_batch, rad_batch):
                    rects_ = []
                    if self.filtering is not None:
                        filter_out = self.filtering(c, rad) 
                    for i in range(len(rad)):
                        c_i = c[i]
                        r_i = rad[i]
                        
                        if not filter_out[i]:
                            rects_.append([c_i, r_i])
                            
                    if len(rects_) == 0:
                        raise Exception("Filtering too strict: no coordinates remain!")
                    self.rects.append(np.array(rects_))
        
        
        if self.verbose > 1:
            rng_rects = tqdm(self.rects)
            rng_rects.set_description("Calculating the CDFs for each of the rects!")
        else:
            rng_rects = self.rects
                
        log_cdfs = []
        if _check_r_primitive():
            for rect in rng_rects:
                diff = norm.cdf(rect[:, 0] + r / rect[:, 1]) - norm.cdf(rect[:, 0] - r / rect[:, 1])
                log_cdfs.append(np.log(diff + self.eps).sum())
        else:
            for r_single, rect in zip(r, rng_rects):
                diff = norm.cdf(rect[:, 0] + r_single / rect[:, 1]) - norm.cdf(rect[:, 0] - r_single / rect[:, 1])
                log_cdfs.append(np.log(diff + self.eps).sum())
                
        return np.array(log_cdfs)
        
    def calculate_mean(self, loader, use_cache=False):
        raise NotImplementedError("You must implement a calculate_mean method for your CDF calculator.")
    

class ParallelogramLogCDF(ParallelogramStatsCalculator):
    def calculate_statistics(self, r, loader, use_cache: bool = False):
        return self.calculate_log_cdf(r, loader, use_cache)

# TODO: implement a convolution-based approach
class GaussianConvolutionStatsCalculator(LatentStatsCalculator):
    pass

class EllipsoidCDFStatsCalculator(LatentStatsCalculator):
    """
    This is a class that is used in methods that use latent ellipsoids.
    For a likelihood model that has an encoding and a decoding into a Gaussian
    space, this ellipsoid calculator yields all the calculations required to 
    calculate the corresponding ellipsoid for each data point.
    
    This class has an abstract encode and decode method that should be implemented for
    specific generative models. For example, for a flow model, that would be running
    the transformation both ways, and for a VAE, that would be running the encoder and
    the decoder.
    
    Moreover, we have computation knobs to optimize the computation of the Jacobian.
    For example, one might use vmaps to optimize the computation of the Jacobian of a 
    batch. Or one can use either functorch or autograd to compute the Jacobian.
    """
    def __init__(
        self,
        *args,
        lim: int = 1000,
        atol: float =1e-4,
        filtering: th.Optional[th.Union[th.Dict[str, str], str]] = None,
        **kwargs,
    ):
        """

        Args:
            lim (int, optional): The limit set for the chi2comb_cdf function. Defaults to 1000.
            atol (float, optional): The tolerance set for chi2comb_cdf function. Defaults to 1e-4.
            filtering: A dypy standard function descriptor. This function
                takes in two torch tensor vectors (cent, rad) and returns a boolean
                mask to filter out a set of corrdinates. For example, the small coordinates might meddle
                with the calculation of the chi2comb_cdf. Or in general, the off-manifold directions
                might be problematic. So, this function can be used to filter out those coordinates.
        """
        super().__init__(*args, **kwargs)
        
        self.lim = lim
        self.atol = atol
        
        if filtering is None:
            self.filtering = lambda cent, rad: torch.zeros_like(rad, dtype=torch.bool)
        elif isinstance(filtering, str):
            self.filtering = dy.eval(filtering)
        elif isinstance(filtering, dict):
            self.filtering = dy.eval_function(**filtering)
        else:
            raise ValueError("filtering should be either a string or a dictionary.")
        
    

    def calculate_single_cdf(self, r, loader, use_cache: bool = False) -> np.ndarray:
        """
        Given two loaders of radii and centers, this function calculates the
        CDF of each of the entailed generalized chi-squared distributions.
        If use_cache is set to true, then this means that centers and radii have been
        cached from before and some of the computations can be optimized.
        
        Args:
            r: Union(float, List[float]) 
                r value(s) where the cdf is evaluated.
            use_cache: (bool)
                Whether to use the cached centers and radii or not.
        Returns:
            np.ndarray: The CDF of each of the ellipsoids when the entire loader
                is flattened. In other words, if batch_i has size sz_i and T = sum(sz_i),
                then the returned array would have size T.
        """
        if not use_cache:
            self.centers, self.radii = calculate_ellipsoids(self, loader)
        if not hasattr(self, 'centers') or not hasattr(self, 'radii'):
            raise ValueError("You should first calculate the ellipsoids before calculating the CDFs with cache.")
        centers, radii = self.centers, self.radii
        def _check_r_primitive():
            if isinstance(r, float):
                return True
            if isinstance(r, int):
                return True
        if self.verbose > 1:
            print("Calculating the single CDFs...")
            if _check_r_primitive():
                print(f"\tr = {r}")
            
        if not use_cache or not hasattr(self, 'chi2s'):
            # if it is not cached, then we should calculate the chi2s
            self.chi2s = []
            for c_batch, rad_batch in zip(centers, radii):
                for c, rad in zip(c_batch, rad_batch):
                    chi2s_ = []
                    if self.filtering is not None:
                        filter_out = self.filtering(c, rad)
                    
                    for i in range(len(rad)):
                        c_i = c[i]
                        r_i = rad[i]
                        if not filter_out[i]:
                            from chi2comb import ChiSquared
                            chi2s_.append(ChiSquared(coef=r_i, ncent=c_i**2, dof=1))
                            
                    if len(chi2s_) == 0:
                        raise Exception("Filtering too strict: no coordinates remain!")
                    self.chi2s.append(chi2s_)
        
        
        if self.verbose > 1:
            rng_chi2s = tqdm(self.chi2s)
            rng_chi2s.set_description("Calculating the CDFs for each of the chi2s")
        else:
            rng_chi2s = self.chi2s
                
        # self.chi2s contains a list of flattened chi2s
        cdfs = []
        if _check_r_primitive():
            for chi2s in rng_chi2s:
                from chi2comb import chi2comb_cdf
                cdfs.append(chi2comb_cdf(r**2, chi2s, 0.0, lim=self.lim, atol=self.atol)[0])
        else:
            for r_single, chi2s in zip(r, rng_chi2s):
                from chi2comb import chi2comb_cdf
                cdfs.append(chi2comb_cdf(r_single**2, chi2s, 0.0, lim=self.lim, atol=self.atol)[0])
                
        return np.array(cdfs)
    
    def calculate_mean(self, loader, use_cache=False):
        if not use_cache:
            self.centers, self.radii = calculate_ellipsoids(self, loader)
        scores = []
        for c_batch, rad_batch in zip(self.centers, self.radii):
            for c, rad in zip(c_batch, rad_batch):
                filter_out = self.filtering(c, rad).numpy()
                # turn filter_out into float
                msk = (1.0 - filter_out.astype(np.float32))
                scores.append(np.sum(msk * rad.numpy() * (1 + c.numpy()**2)))
        return np.array(scores)

    def calculate_statistics(self, r, loader, use_cache: bool = False):
        return self.calculate_single_cdf(r, loader, use_cache)
    
    def sample(self, r, loader, n_samples, use_cache = False):
        if not use_cache:
            self.centers, self.radii, self.rotations = calculate_ellipsoids(self, loader, return_rotations=True, stack_back=False)
        if not hasattr(self, 'centers') or not hasattr(self, 'radii') or not hasattr(self, 'rotations'):
            raise ValueError("You should first calculate the ellipsoids before calculating the CDFs with cache.")
        
        # perform rejection sampling to sample from the ellipsoid according to the latent
        # Gaussian distribution.
        
        ret = []
        # calculate the upper bound of the density values for a Gaussian distribution with 
        # dimesion latent_dim and covariance matrix I
        for cents, radiis, rots in zip(self.centers, self.radii, self.rotations):
            for cent, radii, rot in zip(cents, radiis, rots):
                # turn everything into numpy arrays first
                cent = cent.numpy()
                radii = radii.numpy()
                rot = rot.numpy()
                
                # upper bound for rejection sampling
                d = len(cent)
                M = (2 * np.pi)**(d / 2)
                samples = []
                for _ in range(n_samples):
                    accept = False
                    while not accept:
                        # sample from the latent Gaussian distribution
                        u = np.random.normal(size=len(cent))
                        u = u / np.linalg.norm(u, ord=2)
                        u = u / radii
                        
                        rad = np.power(np.random.uniform(), 1.0 / d) * r
                        # rotate the sample to the ellipsoid frame
                        z = np.matmul(rot.T, u * rad + cent)
                        # calculate the density of the sample
                        prob = np.exp(-0.5 * np.sum(z**2)) / M
                        # sample a uniform value
                        choice = np.random.uniform()
                        # accept the sample if the uniform value is smaller than the density
                        accept = choice < prob
                    
                    # turn z into a torch tensor
                    sample_latent = torch.from_numpy(z).float().device(self.likelihood_model.device).unsqueeze(0)
                    # map back to ambient space
                    sample_ambient = self.encoding_model.decode(sample_latent)
                    samples.append(sample_ambient)
                ret.append(torch.stack(samples).detach().cpu())
                
        return ret
            
             

        