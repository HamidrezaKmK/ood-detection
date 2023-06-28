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
from scipy.special import gamma

class LpBallSampler:
    
    def __init__(
        self,
        # the type of proximity sampling, for example an lp-ball with p=2 samples a sphere with the predefined radius around the center
        p: int = 2,
    ):
        self.p = p
    
    def calculate_norm(self, x):
        return np.linalg.norm(x, ord=self.p, axis=1)
    
    def sample(self, x: th.Tensor, n: int, radius: float) -> th.Tensor:
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
        
        # sample an n by d-1 matrix with values from a normal distribution between -pi and pi
        angles = np.random.uniform(low=-np.pi, high=np.pi, size=(n, d - 1))
        
        # create a matrix sin_mult[i,j] which contains the multiplication of the sin function for angles[i,0], angles[i,1], ..., angles[i,j-1] for j=1,2,...,d-1
        # and for j=0, we have sin_mult[i,0] = 1
        sin_mult = np.concatenate(np.ones((n, 1)), np.cumprod(np.sin(angles), axis=1), axis=1)
        
        # create another matrix called cos_mult[i, j] which is cos(angles[i,j]) for j=0,1,...,d-2 and cos_mult[i, d-1] = 1
        cos_mult = np.concatenate((np.cos(angles), np.ones((n, 1))), axis=1)
        
        coefficients = cos_mult * sin_mult
        # calculate the l-p norm of all the rows of coefficients
        transform_r = radii_samples / self.calculate_norm(coefficients)

        x_s = np.sum(transform_r[:, None] * coefficients, axis=1)
        
        # resize x_s to [n, *shape_of_x] and turn it into a torch tensor with the same device as x, then add x with x_s[i] for i=0,1,...,n-1
        return x + torch.tensor(x_s, device=x.device).reshape(n, *x.shape)
        


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
        
        x: th.Optional[torch.Tensor] = None,
        x_batch: th.Optional[torch.Tensor] = None,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        logger: th.Optional[th.Any] = None,
        
        **kwargs,
    ) -> None:
        super().__init__(x_loader=x_loader, likelihood_model=likelihood_model, logger=logger, **kwargs)
        self.sampler = dy.eval(sampler_cls)(**sampler_args)
        self.radii = np.linspace(l_radius, r_radius, n_radii)
        
        if x is None:
            raise ValueError("x must be provided.")
        
        self.x = x
        self.n_samples = n_samples
        
        if x_batch is not None:
            raise NotImplementedError("x_batch is not implemented yet.")

        if x_loader is not None:
            raise NotImplementedError("x_loader is not implemented yet.")
        
    def run(self):
        """
        Creates a line chart of radius vs score.
        """
        radius_vs_score = []
        for r in self.radii:
            x_prox = self.sampler.sample(self.x, self.n_samples, r)
            all_log_probs = self.likelihood_model.log_prob(x_prox)
            radius_vs_score.append([r, all_log_probs.mean().item()])
        table = wandb.Table(data=radius_vs_score, columns = ["radius", "score"])
        wandb.log(
            {"radius vs score" : wandb.plot.line(table, "radius", "score",
                title="radius vs score")})
            