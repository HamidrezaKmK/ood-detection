from .base_method import OODBaseMethod
import torch
import typing as th
import dypy as dy
import wandb
from tqdm import tqdm
from math import inf
import numpy as np

class HessianSpectrumMonitor(OODBaseMethod):
    """
    This class of methods calculates the derivative of the likelihood model
    log_prob w.r.t the representation, and then calculates the eigenvalues of
    the hessian of the log_prob w.r.t the representation.
    
    After calculating the eigenvalues for each datapoint, it then sorts all the
    eigenvalues and logs a histogram of the eigenvalues.
    
    For multiple datapoints, it calculates eig_mean[k] which is the average k'th largest
    eigenvalue for all the datapoints. It also calculates eig_std[k] which is the standard
    deviation of the k'th largest eigenvalue for all the datapoints.
    """
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        representation_rank: th.Optional[int] = None,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        logger: th.Optional[th.Any] = None,
        data_limit: th.Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(x_loader=x_loader, likelihood_model=likelihood_model, logger=logger, **kwargs)
        self.likelihood_model.denoising_sigma = 0
        self.likelihood_model.dequantize = False
        
        self.representation_rank = representation_rank
        self.data_limit = inf if data_limit is None else data_limit
        if representation_rank is not None:
            raise NotImplementedError("representation_rank is not implemented yet.")

    def run(self):
            
        def nll_func(x):
            return -self.likelihood_model.log_prob(x.unsqueeze(0)).squeeze(0)
        
        eigval_history = None
        device = self.likelihood_model.device
        
        if self.progress_bar:
            iterable = tqdm(range(self.data_limit))
        else: 
            iterable = range(self.data_limit)
            
        # for any index i find the corresponding element in the corresponding batch
        lm = 0
        real_ind = 0
        tt = iter(self.x_loader)
        
        for ind in iterable:
            if ind < lm:
                x = current_batch[real_ind]
            else:
                current_batch, _, _ = next(tt)
                real_ind = ind - lm
                lm += len(current_batch)
                x = current_batch[real_ind]
            real_ind += 1
            
            x = x.to(device)
            # calculate the Hessian w.r.t x0
            
            # TODO: make this compatible with functorch
            hess = torch.autograd.functional.hessian(nll_func, x)

            # Hess has weired dimensions now, we turn it into a 2D matrix
            # by reshaping only the first half.
            # For example, if the shape of hess is (2, 2, 2, 2), then we reshape it to (4, 4)
            # For images of size (1, 28, 28), the shape of hess is (1, 28, 28, 1, 28, 28)
            # and we reshape it to (784, 784)
            hess = hess.squeeze()
            first_half = 1
            for i, dim in enumerate(hess.shape):
                if i < len(hess.shape) // 2:
                    first_half *= dim
            hess = hess.reshape(first_half, -1).detach()
            
            # get all the eigenvalues of the hessian
            eigvals, eigvectors = torch.linalg.eigh(hess)

            # sort all the eigenvalues
            eigvals = eigvals.sort(descending=True).values
            
            
            # detach and put it on cpu and add it to the list
            if eigval_history is None:
                eigval_history = eigvals.detach().cpu().unsqueeze(0)
            else:
                eigval_history = torch.cat([eigval_history, eigvals.detach().cpu().unsqueeze(0)], dim=0)
        
        
        # calculate the mean and std of the k'th largest eigenvalue
        # for all the datapoints using the eigval_history
        eig_mean = torch.mean(eigval_history, dim=0)
        eig_std = torch.std(eigval_history, dim=0)
        
        data = [[mean, std] for mean, std in zip(eig_mean, eig_std)]
        table = wandb.Table(data=data, columns=["eig_mean", "eig_std"])
        wandb.log({'spectrum first order statistics': wandb.plot.histogram(table, "eig_mean",
            title="Mean of the eigenvalues")})
        wandb.log({'spectrum second order statistics': wandb.plot.histogram(table, "eig_std",
            title="STD of the eigenvalues")})