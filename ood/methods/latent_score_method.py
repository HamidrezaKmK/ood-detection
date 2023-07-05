"""
These methods only work on flow models.
They get the latent representation of the data and then compute a score based on that.

Some of the methods just consider the Lp norm of the latent representation of the data.
Some other consider the probability in an Lp ball of radius r around the latent representation of the data.
Others might incorporate semantic information in the local latent space; for example, they
might consider an ellipsoid around the latent representation of the data and then calculate
the probability measure of the ellipsoid. This ellipsoid might be semantically aware with
more variations on the dimensions that are more important for the data.
"""
import typing as th
from typing import Any
import torch
from .base_method import OODBaseMethod
import dypy as dy 
import numpy as np
import wandb
from tqdm import tqdm
from scipy.stats import norm, ncx2
from chi2comb import chi2comb_cdf, ChiSquared
import time
import math

class LpScore:
    """
    Calculates the Lp norm of the latent representation of the data.
    """
    def __init__(self, p: th.Union[float, int, th.Literal['inf']]) -> None:
        self.p = p

    def __call__(self, z: torch.Tensor, x: torch.Tensor, likelihood_model) -> float:
        z = z.cpu().detach().numpy() 
        p = self.p
        if p == 'inf':
            return np.max(np.abs(z), axis=-1)
        else:
            return np.linalg.norm(z, ord=p, axis=-1)
    
class LpProbScore:
    """
    Calculates the probability of the latent representation of the data being in an Lp ball of radius r.
    """
    def __init__(
        self, 
        p: th.Union[float, int, th.Literal['inf']], 
        radius: float,
        eps_correction: float = 1e-6,
    ) -> None:
        self.p = p
        self.radius = radius
        self.eps_correction = eps_correction
    
    def __call__(self, z: torch.Tensor, x: torch.Tensor, likelihood_model) -> float:
        z = z.cpu().detach().numpy() 
        radius = self.radius
        p = self.p
        eps_correction = self.eps_correction
        if p == 'inf':
            log_score = 0
            for z_i in z:
                r = z_i + radius
                l = z_i - radius
                # calculate the standard gaussian CDF of l and r
                cdf_l = norm.cdf(l)
                cdf_r = norm.cdf(r)
                p_i = cdf_r - cdf_l
                p_i = np.clip(p_i, a_min=eps_correction, a_max=1.0)
                log_p_i = np.log(p_i)
                log_score += log_p_i
            return log_score
        else:
            raise NotImplementedError('p != inf not implemented yet!')

    
class SemanticAwareEllipsoidScore:
    """
    This score calculates the latent representation of the data.
    Then considers an allipsoid with it's radii being the magnitude of the 
    jacobian of the latent representation of the data. Then it calculates the 
    integral of the gaussian pdf over the ellipsoid analytically.
    """
    def __init__(
        self, 
        radius: th.Optional[float] = None,
        n_radii: int = 10,
        p: th.Union[float, int, th.Literal['inf']] = 2,
        eps_correction: float = 1e-6,
        use_functorch: bool = True,
        use_forward_mode: bool = True,
    ) -> None:
        self.p = p
        self.radius = radius
        self.eps_correction = eps_correction
        self.use_functorch = use_functorch
        self.use_forward_mode = use_forward_mode
        self.n_radii = n_radii
    
    def __call__(
        self, 
        z: torch.Tensor, 
        x: torch.Tensor, 
        likelihood_model, 
        radius: th.Optional[th.Union[float, np.array]] = None
    ) -> float:
        
        
        for param in likelihood_model.parameters():
            param.requires_grad = False
        
        all_scores = []
        for z_i, x_i in tqdm(list(zip(z, x))):   
            def get_sample_from_latent(z):
                x_samples, logdets = likelihood_model._nflow._transform.inverse(z.unsqueeze(0))
                return x_samples.squeeze(0)
            
            if not self.use_functorch:
                jac = torch.autograd.functional.jacobian(get_sample_from_latent, z_i)
            else:
                if self.use_forward_mode:
                    jac = torch.func.jacfwd(get_sample_from_latent)(z_i)
                else:
                    jac = torch.func.jacrev(get_sample_from_latent)(z_i)
            
            
            jac = jac.reshape((x_i.numel(), z_i.numel())).cpu().detach().numpy()
            
            
            # calculate the eigenspectrum of jac
            eigvals, eigvecs = np.linalg.eigh(jac)
            
            center_rotated = eigvecs.T @ z_i.cpu().numpy()
            
            ellipsoid_radii = np.abs(eigvals) ** self.p
            ellipsoid_radii = ellipsoid_radii / np.sum(ellipsoid_radii)
            
            if self.p == 'inf':
                raise NotImplementedError('p != inf not implemented yet!')
            elif self.p == 2:
                # Dummy test of chi2comb_cdf
                # gcoef = 2
                # ncents = np.abs(center_rotated[:3])
                # q = 1
                # dofs = [1, 1, 1]
                # coefs = ellipsoid_radii[:3]
                # chi2s = [ChiSquared(coefs[i], ncents[i], dofs[i]) for i in range(3)]
                # result, errno, info = chi2comb_cdf(q, chi2s, gcoef)
                # print("Dummy result", result)

                
                chi2s = [ChiSquared(coef=r_i, ncent=c_i**2, dof=1) for c_i, r_i in zip(center_rotated, ellipsoid_radii)]
            
                # check if center_rotated or ellipsoid_radii contains nan or inf
                if np.any(np.isnan(center_rotated)) or np.any(np.isnan(ellipsoid_radii)):
                    raise ValueError("NaN in center_rotated or ellipsoid_radii")
                
                if radius is None:
                    scores = []
                    weights = []
                    for r in np.linspace(0.0, self.radius, self.n_radii):
                        score = chi2comb_cdf(r ** 2, chi2s, 0.0)[0]
                        if 0.05 < score < 0.95:
                            scores.append(math.exp(score))
                            weights.append(math.exp(r))
                    sm = 0.0
                    denom = 0.0
                    for w, s in zip(weights, scores):
                        sm += w * s
                        denom += w
                    score = sm / denom
                    # change this maybe? 
                    score = scores[-1]
                else:
                    score = []
                    for r in radius:
                        score.append(chi2comb_cdf(r ** 2, chi2s, 0.0)[0])
            
            all_scores.append(score)
            
            
        
        for param in likelihood_model.parameters():
            # turn on requires_grad
            param.requires_grad_()
            
            
        return np.array(all_scores)
    
class LatentScore(OODBaseMethod):
    """
    Calculates a score only based on the latent representation of the data.
    
    For example, if we are considering an Lp norm based method and p is given and the
    score is the norm, then it just calculates the norm of the latent representation 
    of the data. According to Nalisnick et al. (2019), the norm for p=2 of the latent 
    representation should fix the pathologies 
    
    TODO: for some reason we can't reproduce for CIFAR10-vs-SVHN
    """
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        
        x: th.Optional[torch.Tensor] = None,
        x_batch: th.Optional[torch.Tensor] = None,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        logger: th.Optional[th.Any] = None,
        #
        score_cls: th.Optional[str] = None,
        score_args: th.Optional[th.Dict[str, th.Any]] = None,
        tunable_parameter_name: th.Optional[str] = None,
        tunable_lr: th.Optional[th.Tuple[float, float]] = None,
        show_std: bool = False,
        # 
        progress_bar: bool = True,
        bincount: int = 5,
        
        **kwargs,
    ) -> None:
        super().__init__(x_loader=x_loader, x=x, x_batch=x_batch, likelihood_model=likelihood_model, logger=logger, **kwargs)
        
        if x is not None:
            self.x_batch = x.unsqueeze(0)
        
        self.progress_bar = progress_bar
        
        self.bincount = bincount
        
        score_args = score_args if score_args is not None else {}

        if score_cls is None:
            raise ValueError("score_cls must be provided!")
        
        self.score_module = dy.eval(score_cls)(**score_args)
        
        self.has_tunable_parameter = False
        if tunable_parameter_name is not None:
            self.tunable_parameter_name = tunable_parameter_name
            self.parameter_values = np.linspace(*tunable_lr, self.bincount)
            self.has_tunable_parameter = True
            self.show_std = show_std
        
            
    def run(self):
        """
        Creates a histogram of scores, with the scores being the lp-norm of the latent representation of the data.
        """
        if not hasattr(self.likelihood_model, '_nflow'):
            raise ValueError('The likelihood model must have a _nflow attribute that returns the number of flows.')
        
        kwargs = {}
        if self.has_tunable_parameter:
            kwargs[self.tunable_parameter_name] = self.parameter_values
            
        all_scores = None
        if self.x_loader is not None:
            if self.progress_bar:
                iterable = tqdm(self.x_loader)
            else:
                iterable = self.x_loader
            for x_batch, _ in iterable:
                with torch.no_grad():
                    z = self.likelihood_model._nflow.transform_to_noise(x_batch)  
                
                new_scores = self.score_module(z, x_batch, self.likelihood_model, **kwargs) 
                all_scores = np.concatenate([all_scores, new_scores], dim=0) if all_scores is not None else new_scores
        else:
            with torch.no_grad():
                z = self.likelihood_model._nflow.transform_to_noise(self.x_batch)   
            all_scores = self.score_module(z, self.x_batch, self.likelihood_model, **kwargs)

        if len(all_scores.shape) > 1:
            mean_scores = []
            mean_minus_std = []
            mean_plus_std = []
            for r, i in zip(self.parameter_values, range(all_scores.shape[1])):
                scores = all_scores[:, i]
                avg_scores = np.mean(scores)
                upper_scores = scores[scores >= avg_scores]
                lower_scores = scores[scores <= avg_scores]
                
                mean_scores.append(avg_scores)
                mean_minus_std.append(avg_scores - np.std(lower_scores))
                mean_plus_std.append(avg_scores + np.std(upper_scores))
            
            if self.show_std:
                wandb.log({
                    "scores": wandb.plot.line_series(
                        xs = self.parameter_values,
                        ys = [mean_scores, mean_minus_std, mean_plus_std],
                        keys = ["mean", "mean-std", "mean+std"],
                        title = "Scores",
                        xname = self.tunable_parameter_name,
                    )
                })
            else:
                wandb.log({
                    "scores": wandb.plot.line_series(
                        xs = self.parameter_values,
                        ys = [mean_scores],
                        keys = ["mean"],
                        title = "Scores",
                        xname = self.tunable_parameter_name,
                    )
                })
        else:    
            # create a density histogram out of all_scores
            # and store it as a line plot in (x_axis, density)
            hist, bin_edges = np.histogram(all_scores, bins=self.bincount, density=True)
            density = hist / np.sum(hist)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # get the average distance between two consecutive centers
            avg_dist = np.mean(np.diff(centers))
            # add two points to the left and right of the histogram
            # to make sure that the plot is not cut off
            centers = np.concatenate([[centers[0] - avg_dist], centers, [centers[-1] + avg_dist]])
            density = np.concatenate([[0], density, [0]])
            
            data = [[x, y] for x, y in zip(centers, density)]
            table = wandb.Table(data=data, columns = ['score', 'density'])
            wandb.log({'score_density': wandb.plot.line(table, 'score', 'density', title='Score density')})