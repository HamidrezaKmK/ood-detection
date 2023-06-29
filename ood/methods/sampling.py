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

class LpBallSampler:
    
    def __init__(
        self,
        # the type of proximity sampling, for example an lp-ball with p=2 samples a sphere with the predefined radius around the center
        p: int = 2,
        accuracy_factor: int = 100,
    ):
        self.p = p
        self.accuracy_factor = accuracy_factor
    
    def calculate_norm(self, x):
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
        
        res = []
        # permute the elements of x and add the noise and then average them out
        x_cum = np.zeros((n, d))
        
        for _ in range(self.accuracy_factor):
            # sample an n by d-1 matrix with values from a normal distribution between -pi and pi
            angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n, d - 1))
            
            # create a matrix sin_mult[i,j] which contains the multiplication of the sin function for angles[i,0], angles[i,1], ..., angles[i,j-1] for j=1,2,...,d-1
            # and for j=0, we have sin_mult[i,0] = 1
            
            def calculate_coefficients(angles):
                sin_mult = np.concatenate((np.ones((n, 1)), np.cumprod(np.sin(angles), axis=1)), axis=1)
                cos_mult = np.concatenate((np.cos(angles), np.ones((n, 1))), axis=1)
                return sin_mult * cos_mult
            
            coefficients = calculate_coefficients(angles)
            # calculate the l-p norm of all the rows of coefficients
            transform_r = radii_samples / self.calculate_norm(coefficients)
            
            x_s = transform_r[:, None] * coefficients
            
            
            # select a random permutation from 1 to d
            perm = np.random.permutation(d)
            
            # permute elements of x_s and add with x_cum
            x_cum += x_s[:, perm]
        
        x_s = x_cum / self.accuracy_factor

        # get the mean and std of x
        x_mean = torch.mean(x.flatten(), dim=0)
        x_std = torch.std(x.flatten(), dim=0)
        
        x_normalized = (x - x_mean) / x_std
        # create a bigger tensor that repeats x_normalized n times
        x_normalized = x_normalized.unsqueeze(0).repeat(n, *[1 for _ in range(len(x_normalized.shape))])
        add_noise = torch.tensor(x_s, device=x.device).reshape(n, *x.shape).type(x.dtype)
        # print("add noise 0", add_noise[0][0])
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
        
        # TODO: plot the maximum, minimum and average norm of the perturbations
        
        
    def run(self):
        """
        Creates a line chart of radius vs score.
        """
        radius_vs_score = []
        if self.progress_bar:
            iterable = tqdm(self.radii)
        else:
            iterable = self.radii
            
        for r in iterable:
            x_prox = self.sampler.sample(self.x, self.n_samples, r)
            all_log_probs = self.likelihood_model.log_prob(x_prox)
            wandb.log({
                'avg_log_prob': torch.mean(all_log_probs).item()
            })
            avg = torch.logsumexp(all_log_probs, dim=0) / self.n_samples
            wandb.log({
                'avg_prob': avg.item()
            })
            radius_vs_score.append([r, avg.item()])
            
            if self.log_image:
                wandb.log(
                    {"proximity samples" : wandb.Image(x_prox[0].cpu().numpy())})
                
        table = wandb.Table(data=radius_vs_score, columns = ["radius", "score"])
        wandb.log(
            {"radius vs score" : wandb.plot.line(table, "radius", "score",
                title="radius vs score")})
    
    
    