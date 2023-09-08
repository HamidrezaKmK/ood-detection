"""
This file contains classes that do statistical analysis in the latent space of a model.
They might also be used for other purposes such as sampling, calculating the convolution
as a proxy for probability volumes, etc.
"""

import torch
import typing as th
import dypy as dy
import numpy as np
from tqdm import tqdm
from .utils import get_device_from_loader
from abc import ABC, abstractmethod

class LatentStatsCalculator(ABC):
    """
    An abstract method used for some statistics calculation on latent encoding models
    such as flows. These models compose an encoding model within and use the Jacobian
    information obtained from the encoding model to do some calculations.
    """
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
    
    @abstractmethod
    def calculate_statistics(self, r, loader, use_cache: bool = False) -> np.ndarray:
        """
        This function calculates some sort of latent statistics w.r.t a radius 'r'
        
        This for example translates to \rho_r(x) values for different inputs in the loader
        or the derivative of that.
        
        use_cache is essential many times. When set to `False`, it performs Jacobian calculations
        from scratch for the loader; however, by setting use_cache=True we are signalling that
        there is no need to recalculate the Jacobians that can be costly.
        
        This especially matters when we have a specific loader and we want to compute the
        statistics for different values of 'r' on the "same" loader.
        """
        raise NotImplementedError("You must implement a calculate_statistics function")
    
    @abstractmethod        
    def get_name(self):
        """An elaborate name for the statistics being calculated"""
        raise NotImplementedError("No get_name function implemented!")
    
    @abstractmethod
    def get_label(self):
        """A short string name for the statistics being calculated"""
        raise NotImplementedError("No get_label function implemented!")
    
class GaussianConvolutionStatsCalculator(LatentStatsCalculator):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
    def get_name(self):
        return "rho_r(x)"
    
    def get_label(self):
        return "convolution"
    
    def _prep_cache(self, loader, use_cache, device):
        """
        This is the actual function that calculates and saves the jacobians and their eigendecompositions.
        """
        
        if not use_cache:
            # get a list of all the Jacobians next to the latent values
            self.jax, self.z_values = self.encoding_model.calculate_jacobian(loader, flatten=True)
            
            # visualize progressbar if verbose > 0
            if self.verbose > 0:
                jax_wrapped = tqdm(self.jax, desc="calculating eigendecomposition of jacobians")
            else:
                jax_wrapped = self.jax
            
            self.jtj_eigvals = []
            self.jtj_eigvecs = []
            
            for j in jax_wrapped:
                j = j.to(device)
                L, Q = torch.linalg.eigh(torch.matmul(j.transpose(1, 2), j))
                L = torch.where(L > 1e-20, L, 1e-20 * torch.ones_like(L))
                
                # move to RAM memory to circumvent overloading the GPU memory
                self.jtj_eigvals.append(L.cpu())
                self.jtj_eigvecs.append(Q.cpu())
        
        # If use_cache is true but the values are not cached at all raise an exception!
        if not hasattr(self, 'jtj_eigvals') or not hasattr(self, 'jtj_eigvecs') or not hasattr(self, 'z_values') or not hasattr(self, 'jax'):
            raise ValueError("You should first calculate the jacobians before calling this function.")
    
    def _calc_score_quant(self, jtj_eigvals, r, jtj_rot, jac, z_0):
        """
        Given a single jacobian alongside the eigehvalues of J^TJ and the rotations,
        computes the latent statistical value itself. This function is internally used in calculate_statistics,
        but isolates the actual methametical computations per-datapoint.
        """
        if isinstance(r, torch.Tensor):
            var = torch.exp(2 * r)
        else:
            var = np.exp(2 * r)
        
        d = len(z_0)
        log_pdf = -0.5 * d * np.log(2 * np.pi)
        log_pdf = log_pdf - 0.5 * torch.sum(torch.log(jtj_eigvals + var))
        z_ = (jtj_rot.T @ z_0.reshape(-1, 1)).reshape(-1)
        log_pdf = log_pdf - torch.sum(jtj_eigvals * z_ * z_ / (jtj_eigvals + var)) / 2
        
        return log_pdf
    
                    
    def calculate_statistics(self, r, loader, use_cache: bool = False, **kwargs) -> np.ndarray:
        """
        This is the actual implementation of the rho_r based on our formulations.
        
        If use_cache is set to False, we first compute the Jacobians of the entire loader based
        on the encoding_model. Then we only keep the eigenvalues and rotations of J^TJ for each of 
        the Jacobians, because that is the only information needed from the Jacobians to calculate 
        the rho_r value for "small" 'r'. 
        """
        
        # get the device of the loader for potential cuda computation
        device = get_device_from_loader(loader)
        
        # get the jacobians and do any pre-computation needed
        self._prep_cache(loader, use_cache, device)

        
        latent_values = []
        
        for x_batch, jtj_eigvals_batch, jtj_eigvecs_batch, jac_batch, z_batch in zip(loader, self.jtj_eigvals, self.jtj_eigvecs, self.jax, self.z_values):
            
            for x_0, jtj_eigvals, jtj_rot, jac, z_0 in zip(x_batch, jtj_eigvals_batch, jtj_eigvecs_batch, jac_batch, z_batch):
                # compute the log density of a standard Gaussian distribution
                # with mean zero and covariance matrix (J @ J.T + r * I)
                
                x_0 = x_0.to(device)
                jtj_eigvals = jtj_eigvals.to(device)
                jtj_rot = jtj_rot.to(device)
                jac = jac.to(device)
                z_0 = z_0.to(device)
                
                # end up computing log density of N(x_0 - J . z_0, J J^T + r.I) (x0)
                val = self._calc_score_quant(jtj_eigvals, r, jtj_rot, jac, z_0, **kwargs)
                # if val is a tensor of size 1, then we should convert it to a float
                if isinstance(val, torch.Tensor):
                    val = val.cpu().item()
                latent_values.append(val)
                    
        return np.array(latent_values)
    
    def sample(
        self, 
        r, 
        loader, 
        n_samples, 
        use_cache = False, 
    ):
        cur = loader
        if not isinstance(cur, list):
            cur = next(iter(cur))
        else:
            while isinstance(cur, list):
                cur = cur[0]
        device = cur.device
        
        self._prep_cache(loader, use_cache, device) 
        
        ret = []
        # calculate the upper bound of the density values for a Gaussian distribution with 
        # dimesion latent_dim and covariance matrix I
        
        if self.verbose >= 1:
            outer_range = tqdm(zip(loader, self.jax, self.z_values), desc="sampling from latent gaussians", total=len(loader))
        else:
            outer_range = zip(loader, self.jax, self.z_values)
        
        idx2 = 0
        for x_batch, jac_batch, z_batch in outer_range:
            idx2 += 1
            idx = 0
            fail_cnt = 0
            for x, jac, z in zip(x_batch, jac_batch, z_batch):
                
                idx += 1
                x = x.to(device)
                jac = jac.to(device)
                z = z.to(device)

                
                # calculate the pseudo-inverse of jac @ jac^T
                # L, Q = torch.linalg.eigh(jac.T @ jac)
                
                # covariance_matrix = max(r, 1e-3) * Q @ torch.diag(1 / L) @ Q.T
                
                # sample from a multivariate_normal distribution
                # with mean zero and covariance matrix covariance_matrix
                covariance_matrix = r * torch.linalg.pinv(jac.T @ jac)
                L, Q = torch.linalg.eigh(covariance_matrix)
                L = torch.where(L > 1e-4, L, 1e-4 * torch.ones_like(L))
                covariance_matrix = Q @ torch.diag(L) @ Q.T
                
                try:
                    all_samples = torch.distributions.MultivariateNormal(
                        loc = torch.zeros_like(z),
                        covariance_matrix=covariance_matrix,
                    ).rsample((n_samples,))
                except ValueError as e:
                    all_samples = torch.zeros_like(z).repeat(n_samples, *[1 for _ in range(len(z.shape))])
                    min_eig = torch.linalg.eigvalsh(covariance_matrix).min()
                    fail_cnt += 1
                    if min_eig > 1e-10:
                        print("saw strange minimum:", min_eig.item())
                    
                if self.verbose >= 1:
                    outer_range.set_description(f"sampling from latent gaussians - fail [{fail_cnt}/{len(x_batch)}]")
                    
                all_perturbed_z = z[None, :] + all_samples
                all_x_perturbed = self.encoding_model.decode(all_perturbed_z, batchwise=True)
                if not self.encoding_model.diff_transform:
                    all_x_perturbed = self.likelihood_model._inverse_data_transform(all_x_perturbed)
                ret.append(all_x_perturbed.detach().cpu())
        return ret
         

class GaussianConvolutionRateStatsCalculator(GaussianConvolutionStatsCalculator):
    """
    This class computes the i'th order gradients of the approximated likelihood which is the proxy
    for the intrinsic dimension itself.
    """
    
    def get_name(self):
        return "deriv_rho_r(x)"
    
    def get_label(self):
        return "fast-LID"
    
    def _calc_score_quant(
        self, 
        jtj_eigvals, 
        r, 
        jtj_rot, 
        jac, 
        z_0, 
    ):
        
        if isinstance(r, torch.Tensor):
            var = torch.exp(2 * r)
        else:
            var = np.exp(2 * r)
    
        ret = - torch.sum(1 / (jtj_eigvals + var))
        z_ = (jtj_rot.T @ z_0.reshape(-1, 1)).reshape(-1)
        ret = ret + torch.sum(jtj_eigvals * (z_ / (jtj_eigvals + var)) ** 2)

        return ret * var
      