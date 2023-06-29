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
    
    def sample(self, x: torch.Tensor, n: int, radius: float) -> torch.Tensor:
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
        return x_perturbed * x_std + x_mean
        


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
        n_samples: int = 100,
        
        sampler_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        # 
        l_radius: float = 0.1,
        r_radius: float = 0.5,
        n_radii: int = 10,
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
        self.radii = np.linspace(l_radius, r_radius, n_radii)
        
        if radii is not None:
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
        
        int_radii = [int(r * mx_divided_by_mn_radii) for r in self.radii]
        
        if self.progress_bar:
            iterable = tqdm(list(zip(self.radii, int_radii)))
        else:
            iterable = list(zip(self.radii, int_radii))
        
        cnt = 0
        for x in iterable:
            r, idx = x
            x_prox = self.sampler.sample(self.x, self.n_samples, r)
            all_log_probs = self.likelihood_model.log_prob(x_prox)
            wandb.log({
                'avg_log_prob': torch.mean(all_log_probs).item()
            }, step=idx)
            log_avg = torch.logsumexp(all_log_probs, dim=0) - math.log(self.n_samples)
            wandb.log({
                'log_P': log_avg.item()
            }, step=idx)
            radius_vs_score.append([r, log_avg.item()])
            
            if self.log_image and (cnt + 1) % self.img_log_freq == 0:
                wandb.log(
                    {"proximity samples" : wandb.Image(x_prox[0].cpu().numpy())}, step=idx)
            cnt += 1
            
        table = wandb.Table(data=radius_vs_score, columns = ["radius", "score"])
        wandb.log(
            {"radius vs score" : wandb.plot.line(table, "radius", "score",
                title="radius vs score")})
    
    
    