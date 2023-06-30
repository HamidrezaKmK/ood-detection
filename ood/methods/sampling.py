"""
This codes relate to sampling techniques for determining OOD data.
The general idea is that DGMs are capable of generating visibly good in-distribution data, but when they are compared w.r.t 
their likelihood scores, they are not able to distinguish between in-distribution and OOD data.
The discrepency between likelihood scores and probabilities can only be explained through peaky distributions.
Therefore, instead of calculating a point score as a likelihood, we sample points that centered around a specific point of interest.
These might be circumscribed by an L2 ball or any other shape. Then, when we average it out, we get a score that is more representative
of the probabilities, rather than mere likelihoods.
"""

import typing as th
import torch
from .base_method import OODBaseMethod
import dypy as dy
import numpy as np
import wandb
from tqdm import tqdm
import math
from scipy.stats import gennorm, norm, laplace, uniform
from scipy.special import gammaln

class LpBallSampler:
    
    def __init__(
        self,
        # the type of proximity sampling, for example an lp-ball with p=2 samples a sphere with the predefined radius around the center
        p: th.Union[int, float, th.Literal['inf']] = 2,
        accuracy_factor: int = 100,
    ):
        self.p = p
        self.accuracy_factor = accuracy_factor
    
    def calculate_norm(self, x):
        if self.p == 'inf':
            return np.max(np.abs(x), axis=1)
        else:
            return np.linalg.norm(x, ord=self.p, axis=1)
    
    def sample(self, x: torch.Tensor, n: int, radius: float, likelihood_model: torch.nn.Module) -> torch.Tensor:
        # we use the following volume function to calculate the volume of the lp-ball
        # V = 2^d * r^d * gamma(1 + 1/p)^d / gamma(1 + d/p)
        # we know this can represent the CDF of the radius of the lp-ball
        # CDF = V(r) / V(R) = r^d / R^d
        # therefore, for sampling, we may calculate the inverse of this function w.r.t r
        # such that given a random variable u ~ U(0, 1), we can calculate the radius of the lp-ball
        
        # generate n uniform random variables
        u = np.random.uniform(size=n)
        
        # invert them to get the radii
        d = x.numel()
        
        radii_samples = np.power(u, 1.0 / d) * radius
        
        if self.p == 'inf':
            x_s = uniform.rvs(loc=-1, scale=2, size=(n, d))
            x_s = x_s / np.linalg.norm(x_s, ord=np.inf, axis=1)[:, None]
        elif abs(self.p - 2) < 1e-6:
            x_s = norm.rvs(size=(n, d))
            x_s = x_s / np.linalg.norm(x_s, ord=2, axis=1)[:, None]
        elif abs(self.p - 1) < 1e-6:
            x_s = laplace.rvs(size=(n, d))
            x_s = x_s / np.linalg.norm(x_s, ord=1, axis=1)[:, None]
        else:
            x_s = gennorm.rvs(beta=self.p, size=(n, d))
            x_s = x_s / np.linalg.norm(x_s, ord=self.p, axis=1)[:, None]
        
        x_s = x_s * radii_samples[:, None]
        
        # get the mean and std of x
        x_mean = torch.mean(x.flatten(), dim=0)
        x_std = torch.std(x.flatten(), dim=0)
        
        x_normalized = (x - x_mean) / x_std
        # create a bigger tensor that repeats x_normalized n times
        x_normalized = x_normalized.unsqueeze(0).repeat(n, *[1 for _ in range(len(x_normalized.shape))])
        add_noise = torch.tensor(x_s, device=x.device).reshape(n, *x.shape).type(x.dtype)
        x_perturbed = x_normalized + add_noise
        x_prox = x_perturbed * x_std + x_mean
        with torch.no_grad():
            all_log_probs = likelihood_model.log_prob(x_prox).detach()

        if self.p != 'inf':
            log_vol = d * math.log(radius + 1e-6) 
            log_vol += d * math.log(2.0) 
            log_vol += d * gammaln(1 + 1 / self.p)            
            log_vol -= gammaln(1 + d / self.p)

        else:
            log_vol = d * (math.log(radius + 1e-6) + math.log(2))
        
        return (
            x_prox, 
            (torch.logsumexp(all_log_probs, dim=0) - math.log(n) + log_vol).item(),
            torch.mean(all_log_probs).item() + log_vol,
            torch.std(all_log_probs).item() + log_vol,
        )   


class LpLatentFlowBallSampler:
    
    def __init__(
        self,
        # the type of proximity sampling, for example an lp-ball with p=2 samples a sphere with the predefined radius around the center
        p: th.Union[int, float, th.Literal['inf']] = 2,
        eps_correction: float = 1e-20,
    ):
        self.p = p
        if self.p != 'inf':
            raise NotImplementedError('Only Lp ball with p=inf is supported for now.')

        self.eps_correction = eps_correction
    
    def calculate_norm(self, x):
        if self.p == 'inf':
            return np.max(np.abs(x), axis=1)
        else:
            return np.linalg.norm(x, ord=self.p, axis=1)
    
    def sample(self, x: torch.Tensor, n: int, radius: float, likelihood_model: torch.nn.Module) -> torch.Tensor:
        
        if not hasattr(likelihood_model, '_nflow'):
            raise ValueError('The likelihood model must have a _nflow attribute that returns the number of flows.')
        with torch.no_grad():
            z = likelihood_model._nflow.transform_to_noise(x.unsqueeze(0)).detach().squeeze(0)
            
        log_P = 0.0
        all_sampled_z = []
        for z_i_device in z:
            z_i = z_i_device.cpu().item()
            r = z_i + radius
            l = z_i - radius
            # calculate the standard gaussian CDF of l and r
            cdf_l = norm.cdf(l)
            cdf_r = norm.cdf(r)
            p_i = cdf_r - cdf_l
            p_i = np.clip(p_i, a_min=self.eps_correction, a_max=1.0)
            log_p_i = np.log(p_i)
            
            log_P += log_p_i
            # sample uniformly from the [l, r] interval
            u = np.random.uniform(size=n) * (r-l) + l
            
            u = torch.from_numpy(u).to(z_i_device.device).type(z_i_device.dtype)
            all_sampled_z.append(u)
            
        all_sampled_z = torch.stack(all_sampled_z).transpose(0, 1)
        
        x_prox = None
        if hasattr(likelihood_model._nflow, '_transform'):
            x_prox, _ = likelihood_model._nflow._transform.inverse(all_sampled_z)
        
        with torch.no_grad():
            all_log_probs = likelihood_model.log_prob(x_prox).detach()
            
        # returns samples, log_P, 
        return (
            x_prox, 
            log_P,
            torch.mean(all_log_probs).item(),
            torch.std(all_log_probs).item(),
        )
    
class SamplingOODDetection(OODBaseMethod):
    """
    This class of methods that relies on sampling proximity points to get a score
    that is more representative of the probability of the data, rather than the likelihood.
    """
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        
        #
        sampler_cls: str,
        n_samples: int,
        
        sampler_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        # 
        l_radius: float = 0.1,
        r_radius: float = 0.5,
        n_radii: int = 10,
        scaling_radii: th.Literal['linear', 'log', 'rev_log'] = 'linear',
        radii: th.Optional[th.List] = None,
        
        x: th.Optional[torch.Tensor] = None,
        x_batch: th.Optional[torch.Tensor] = None,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        logger: th.Optional[th.Any] = None,
        
        # 
        progress_bar: bool = True,
        log_image: bool = True,
        img_log_freq: int = 10,
        
        **kwargs,
    ) -> None:
        super().__init__(x_loader=x_loader, x=x, x_batch=x_batch, likelihood_model=likelihood_model, logger=logger, **kwargs)
        sampler_args = sampler_args or {}
        self.sampler = dy.eval(sampler_cls)(**sampler_args)
        
        # create a sequence of n_samples radii between l_radius and r_radius
        # such that in the beginning, we have a lot of radii, and then we have less and less
        if scaling_radii == 'linear':
            self.radii = np.linspace(l_radius, r_radius, n_radii)
        elif scaling_radii == 'log':
            self.radii = np.logspace(np.log10(l_radius + 1e-6), np.log10(r_radius), n_radii - 1)
            # add l_radius to the beginning
            self.radii = np.concatenate([np.array([l_radius]), self.radii])
        elif scaling_radii == 'rev_log':
            self.radii = r_radius - np.concatenate([np.array([0.0]), np.logspace(np.log10(1e-6), np.log10(r_radius - l_radius), n_radii - 1)])
            # reverse the order
            self.radii = self.radii[::-1]
        else:
            raise ValueError(f"scaling_radii must be either 'linear' or 'log', but got {scaling_radii}.")
        self.use_rescaling = False
        
        if radii is not None:
            self.use_rescaling = True
            self.radii = np.array(radii).astype(np.float32)
        
        if x is None:
            raise ValueError("x must be provided.")
        
        self.x = x
        self.n_samples = n_samples
        
        if x_batch is not None:
            raise NotImplementedError("x_batch is not implemented yet.")

        if x_loader is not None:
            raise NotImplementedError("x_loader is not implemented yet.")
        
        self.progress_bar = progress_bar
        self.log_image = log_image
        self.img_log_freq = img_log_freq
        
        # TODO: plot the maximum, minimum and average norm of the perturbations
        
        
    def run(self):
        """
        Creates a line chart of radius vs score.
        The scores are either the log of the probability of being in that ball, or it is the average of the log probs in that ball
        """
        radius_vs_score = []
        
        first_non_zero = None
        for r in self.radii:
            if abs(r) > 1e-6:
                first_non_zero = r
                break
            
        mx_divided_by_mn_radii = self.radii[-1] / first_non_zero
        
        int_radii = [None if not self.use_rescaling else int(r * mx_divided_by_mn_radii) for r in self.radii]
        
        if self.progress_bar:
            iterable = tqdm(list(zip(self.radii, int_radii)))
        else:
            iterable = list(zip(self.radii, int_radii))
        
        cnt = 0
        for x in iterable:
            r, idx = x
            
            x_prox, log_P, avg_log_prob, std_log_prob = self.sampler.sample(self.x, self.n_samples, r, likelihood_model=self.likelihood_model)
            wandb.log({
                'radii': r
            })
            wandb.log({
                'avg_log_prob': avg_log_prob
            })
            wandb.log({
                'std_log_prob': std_log_prob
            })
            wandb.log({
                'log_P': log_P
            })
            wandb.log({
                'P': np.exp(log_P)
            })
            radius_vs_score.append([r, log_P])
            
            if self.log_image and cnt % self.img_log_freq == 0 and x_prox is not None:
                wandb.log(
                    {"proximity/sample" : wandb.Image(x_prox[0].cpu())})
                wandb.log({
                    "proximity/average": wandb.Image(torch.mean(x_prox, dim=0).cpu())
                })
                
                # for every pixel in every channel, calculate the difference between
                # that pixel and the average of the surronding 3 x 3 pixels
                
                # calculate avg_x_prox[c,i,j] where it is equal to the average of the 3 x 3 pixels
                avg_x_prox = torch.zeros_like(x_prox)
                avg_x_prox = avg_x_prox.unfold(2, 3, 1).unfold(3, 3, 1).mean(dim=(4,5))
                
            cnt += 1
            
        table = wandb.Table(data=radius_vs_score, columns = ["radius", "log_P"])
        wandb.log(
            {"radius vs prob" : wandb.plot.line(table, "radius", "log_P",
                title="radius vs prob")})
    
    
    