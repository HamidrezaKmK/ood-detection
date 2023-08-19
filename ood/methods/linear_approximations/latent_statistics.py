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
from scipy.stats import multivariate_normal

def _check_primitive(r: th.Any):
    """
    A simple function that checks if an input type is numerical or not
    """
    if isinstance(r, float):
        return True
    if isinstance(r, int):
        return True
    return False

class LatentStatsCalculator:
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        # Encoding and decoding model
        encoding_model_class: th.Optional[th.Union[str, type]] = None, 
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
        if self.verbose > 1:
            print("Calculating the single CDFs...")
            if _check_primitive(r):
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
        if _check_primitive(r):
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
    
    def get_name(self):
        return "P(parallelogram)"

    def get_label(self):
        return "log-CDF"
    
    def calculate_statistics(self, r, loader, use_cache: bool = False):
        return self.calculate_log_cdf(r, loader, use_cache)


# TODO: implement a convolution-based approach
class GaussianConvolutionStatsCalculator(LatentStatsCalculator):
    def __init__(
        self,
        *args,
        acceleration: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.acceleration = acceleration
        
    def get_name(self):
        return "[N(0, r . I) * p_theta] (x)"
    
    def get_label(self):
        return "convolution"
    
    def calculate_statistics(self, r, loader, use_cache: bool = False) -> np.ndarray:
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
        # get the device from loader
        # NOTE: This is jankey, but it works for now
        cur = loader
        while isinstance(cur, list):
            cur = cur[0]
        device = cur.device
        
        if not use_cache:
            self.jax, self.z_values = self.encoding_model.calculate_jacobian(loader, stack_back=True, flatten=True)
            
            if self.verbose > 0:
                rng = tqdm(self.jax, desc="calculating eigendecomposition of jacobians")
            else:
                rng = self.jax
            self.jjt_eigs = []
            for j in rng:
                if self.acceleration > 0:
                    j = j.to(device)
                    self.jjt_eigs.append(torch.linalg.eigvalsh(torch.matmul(j, j.transpose(1, 2))).cpu())
                else:
                    self.jjt_eigs.append([0 for _ in range(j.shape[0])])
                    
        if not hasattr(self, 'jjt_eigs') or not hasattr(self, 'z_values') or not hasattr(self, 'jax'):
            raise ValueError("You should first calculate the jacobians before calling this function.")
            
        
        # if it is not cached, then we should calculate the chi2s
        conv = []
        cumul_l = 0
        cumul_r = 0
        
        for x_batch, jjt_eigs_batch, jac_batch, z_batch in zip(loader, self.jjt_eigs, self.jax, self.z_values):
            
            # cumul_l and cumul_r are the cumulative left and right indices
            cumul_r += len(x_batch)
            if _check_primitive(r):
                r_ = np.repeat(r, len(x_batch))
            else:
                r_ = r[cumul_l:cumul_r]
            cumul_l += len(x_batch)
            cumul_r += len(x_batch)
                
            for x, jjt_eigs, J, z, r in zip(x_batch, jjt_eigs_batch, jac_batch, z_batch, r_):
                # compute the log density of a standard Gaussian distribution
                # with mean zero and covariance matrix (J @ J.T + r * I)
                
                x = x.to(device)
                jjt_eigs = jjt_eigs.to(device)
                J = J.to(device)
                z = z.to(device)
                
                if self.acceleration >= 3:
                    d = len(z)
                    log_pdf = -0.5 * d * np.log(2 * np.pi)
                    log_pdf += - 0.5 * torch.sum(torch.log(jjt_eigs + r))
                    conv.append(log_pdf.cpu().item())
                elif self.acceleration >= 2:
                    # implement using torch.distributions with the multivariate normal density function
                    pass
                elif self.acceleration >= 1:
                    from torch.distributions.multivariate_normal import MultivariateNormal
                    # x = x.flatten()
                    # mean = (x - J @ z.unsqueeze(-1)).squeeze()
                    # print("HI before!")
                    d = len(z)
                    cov = J @ J.T + r * torch.eye(J.shape[0], device=device) # Must be a positive-definite matrix
                    log_pdf = -0.5 * d * np.log(2 * np.pi)
                    log_pdf += - 0.5 * torch.logdet(cov)
                    conv.append(log_pdf.cpu().item())
                    
                    # distribution = MultivariateNormal(mean, covariance_matrix=cov)
                    # log_pdf_value = distribution.log_prob(x)
                    # conv.append(log_pdf_value.cpu().item())
                elif self.acceleration >= 0:
                    J = J.cpu().numpy()
                    x = x.cpu().numpy().reshape(-1, 1)
                    z = z.cpu().numpy().reshape(-1, 1)
                    d = J.shape[0]
                    
                    conv.append(multivariate_normal.logpdf(x, mean=(x - J @ z).reshape(-1), cov=J @ J.T + r * np.eye(d)))
                
        return np.array(conv)


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
        
    def get_name(self):
        return "P(R^2 < r^2)"

    def get_label(self):
        return "CDF"
    
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
        
        if self.verbose > 1:
            print("Calculating the single CDFs...")
            if _check_primitive(r):
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
        if _check_primitive(r):
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
    
    def sample(
        self, 
        r, 
        loader, 
        n_samples, 
        use_cache = False, 
        rejection_sampling = False, 
        disregard_radii: bool = False
    ):
        if not use_cache:
            self.centers, self.radii, self.rotations = calculate_ellipsoids(self, loader, return_rotations=True, stack_back=False)
        if not hasattr(self, 'centers') or not hasattr(self, 'radii') or not hasattr(self, 'rotations'):
            raise ValueError("You should first calculate the ellipsoids before calculating the CDFs with cache.")
        
        # perform rejection sampling to sample from the ellipsoid according to the latent
        # Gaussian distribution.
        
        ret = []
        # calculate the upper bound of the density values for a Gaussian distribution with 
        # dimesion latent_dim and covariance matrix I
        idx = 0
        
        if self.verbose > 0:
            outer_range = tqdm(zip(self.centers, self.radii, self.rotations), desc="sampling from ellipsoids", total=len(self.centers))
        else:
            outer_range = zip(self.centers, self.radii, self.rotations)
            
        for cents, radiis, rots in outer_range:
            for cent, radii, rot in zip(cents, radiis, rots):
                idx += 1
                # turn everything into numpy arrays first
                cent = cent.numpy()
                radii = radii.numpy()
                radii = np.where(radii > 0, radii, 1e-6)
                rot = rot.numpy()
                
                # upper bound for rejection sampling
                d = len(cent)
                samples = []
                if self.verbose > 1:
                    rng = tqdm(range(n_samples), desc=f"sampling from batch {idx}")
                else:
                    rng = range(n_samples)
                    
                for _ in rng:
                    accept = False
                    try_count = 0
                    while not accept:
                        if rejection_sampling:
                            try_count += 1
                            rng.set_description(f"rejection sampling from batch {idx} (try {try_count})")
                        
                        # get a sample from a multivariate normal 
                        # and reduce it to the level set of the ellipsoid
                        u = np.random.normal(size=d)
                        u = u / np.linalg.norm(u, ord=2)
                        if not disregard_radii:
                            u = u / radii
                        
                        # get the radius (this would most probably be on the perimeter with rad=r
                        # due to high dimensionality)
                        rad = np.power(np.random.uniform(), 1.0 / d) * r
                        # rotate the sample back
                        z = np.matmul(rot.T, u * rad + cent)
                        
                        if not rejection_sampling:
                            accept = True
                        else:
                            # calculate the density of the sample
                            log_prob_sample = -0.5 * np.sum(z**2) - (d/2) * np.log(2 * np.pi)
                            tt = np.where(np.abs(cent) - r > 0, np.abs(cent) - r, 0)
                            log_density_upper_bound = -0.5 * np.sum(tt**2) - (d/2) * np.log(2 * np.pi)
                            
                            log_P_accept = log_prob_sample - log_density_upper_bound
                            log_choice = np.log(np.random.uniform())
                            # accept the sample if the uniform value is smaller than the density
                            
                            accept = log_choice < log_P_accept
                    
                    # turn z into a torch tensor
                    sample_latent = torch.from_numpy(z).float()
                    sample_latent = sample_latent.to(self.likelihood_model.device).unsqueeze(0)
                    
                    # map back to ambient space
                    sample_ambient = self.encoding_model.decode(sample_latent).squeeze(0)
                    samples.append(self.likelihood_model._inverse_data_transform(sample_ambient).detach().cpu())
                ret.append(torch.stack(samples))
                
        return ret
            
             

class FirstMomentEllipsoidsStatsCalculator(EllipsoidCDFStatsCalculator):
    
    def calculate_statistics(self, r, loader, use_cache: bool = False):
        return self.calculate_mean(loader, use_cache)