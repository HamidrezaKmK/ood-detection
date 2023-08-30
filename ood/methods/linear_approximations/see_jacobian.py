"""
This ood detection method simply prints out the trend of the eigenspectrum of the OOD 
data loader. The hope is, that there are certain anomalies one can observe in the 
Jacobian of OOD data that can be isolated.
"""  
import torch
from ood.methods.base_method import OODBaseMethod
import typing as th
from .encoding_model import EncodingModel
import dypy as dy
from .utils import buffer_loader
import numpy as np
from ood.visualization import visualize_trends, visualize_scatterplots
from tqdm import tqdm

class SeeJacobian(OODBaseMethod):
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        # 
        x: th.Optional[torch.Tensor] = None,
        x_batch: th.Optional[torch.Tensor] = None,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        logger: th.Optional[th.Any] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        
        # jacobian calculator
        encoding_model_class: th.Optional[str] = None,
        encoding_model_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        
        verbose: int = 0,
        
        chunk_size: int = 1,
        visualization_args: th.Optional[th.Dict[str, th.Any]] = None,
        discard_quantile: th.Optional[float] = None,
    ):
        """
        Args:
            likelihood_model (torch.nn.Module): The likelihood model with initialized parameters that work best for generation.
            x (th.Optional[torch.Tensor], optional): If the OOD detection task is only aimed for single point, this is the point.
            x_batch (th.Optional[torch.Tensor], optional): If the OOD detection task is aimed for a single batch.
            x_loader (th.Optional[torch.utils.data.DataLoader], optional): If the OOD detection methods is to be applied on the entire
                dataloader.
            logger (th.Optional[th.Any], optional): This is the logger being used.
            in_distr_loader (th.Optional[torch.utils.data.DataLoader], optional): This is the dataloader pertaining to the training data,
                in OOD detection, we have access to the training data itself and our method might do some relevant computations on that.
            verbose (int, optional): This controls the logging in our model:   
                (0) No logging
                (1) Partial logging
                (2) Many logs
            latent_statistics_calculator_class: (str) The class of the latent statistics calculator
            latent_statistics_calculator_args (dict) The arguments used for visualizing the latent statistics calculator
        """
        super().__init__(
            x_loader=x_loader, 
            x=x, 
            x_batch=x_batch, 
            likelihood_model=likelihood_model, 
            logger=logger, 
            in_distr_loader=in_distr_loader, 
        )
        
        encoding_model_args = encoding_model_args or {}
            
        self.encoding_model: EncodingModel = dy.eval(encoding_model_class)(likelihood_model, **encoding_model_args)
        
        if self.x is not None:    
            self.x_batch = self.x.unsqueeze(0)
        
        if self.x_batch is not None:
            # create a loader with that single batch
            self.x_loader = [(self.x_batch, None, None)]
        
        self.verbose = verbose
        
        self.visualization_args = visualization_args or {}
        self.chunk_size = chunk_size
        
        cur = self.x_loader
        if not isinstance(cur, list):
            cur = next(iter(cur))
        else:
            while isinstance(cur, list):
                cur = cur[0]
        self.device = cur.device
        D = cur.numel() // cur.shape[0]
        
        if discard_quantile is None:
            self.keep_rank = 0
        else:
            self.keep_rank = int(discard_quantile * D)
        
    def run(self):
    
        jax, _ = self.encoding_model.calculate_jacobian(
            loader=self.x_loader,
        )
        
        
        
        all_values = None
        if self.verbose > 0:
            jax = tqdm(jax, desc="Calculating the singular values")
        
        for jac_batch in jax:
            
            l = 0
            while l < jac_batch.shape[0]:
                if self.verbose > 0:
                    jax.set_description(f"Calculating SVD of chunk [{l // self.chunk_size + 1}/{(jac_batch.shape[0] + self.chunk_size - 1) // self.chunk_size}]")
                r = min(l + self.chunk_size, jac_batch.shape[0])
                jac_chunk = jac_batch[l:r].to(self.device)
                singular_values_chunk = torch.linalg.svdvals(jac_chunk)
                
                singular_values_chunk = singular_values_chunk.cpu().numpy()
                all_values = np.concatenate([all_values, singular_values_chunk], axis=0) if all_values is not None else singular_values_chunk
                l += self.chunk_size
                
                
                
        all_values = all_values[:, self.keep_rank:]
        
        visualize_trends(
            scores=all_values,
            t_values=np.arange(all_values.shape[1]),
            reference_scores=None,
            x_label="idx_eigenvalue",
            y_label="singular-values",
            title="The singular value spectrum in descending order",
            **self.visualization_args,
        )
        

class IntrinsicDimensionFixing(SeeJacobian):
    def __init__(
        self,
        *args,
        singular_value_threshold: float = 0.0015,
        poly_deg: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.likelihood_model = self.likelihood_model.to(self.device)
        self.singular_value_threshold = singular_value_threshold
        self.poly_deg = poly_deg
    
    def run(self):
    
        jax, z_values = self.encoding_model.calculate_jacobian(
            loader=self.x_loader,
        )
        
        all_dimensionality_scores = None
        all_likelihoods = None
        
        if self.verbose > 0:
            jax = tqdm(jax, desc="Calculating the singular values")
        
        for jac_batch, z_batch in zip(jax, z_values):
        
            l = 0
            while l < jac_batch.shape[0]:
                if self.verbose > 0:
                    jax.set_description(f"Calculating SVD of chunk [{l // self.chunk_size + 1}/{(jac_batch.shape[0] + self.chunk_size - 1) // self.chunk_size}]")
                r = min(l + self.chunk_size, jac_batch.shape[0])
                jac_chunk = jac_batch[l:r].to(self.device)
                singular_values_chunk = torch.linalg.svdvals(jac_chunk)
                
                singular_values_chunk = singular_values_chunk.cpu().numpy()
                singular_values_chunk = np.where(singular_values_chunk < self.singular_value_threshold, singular_values_chunk, 0)
                
                new_dimensionality_scores = np.sum(singular_values_chunk, axis=-1).flatten()
                all_dimensionality_scores = np.concatenate([all_dimensionality_scores, new_dimensionality_scores]) if all_dimensionality_scores is not None else new_dimensionality_scores
            
                l += self.chunk_size
            
            z_batch = z_batch.to(self.device)
            x_batch = self.encoding_model.decode(z_batch, batchwise=True)
            
            with torch.no_grad():
                likelihoods = self.likelihood_model.log_prob(x_batch).cpu().numpy().flatten()
                likelihoods = likelihoods ** self.poly_deg
                all_likelihoods = np.concatenate([all_likelihoods, likelihoods]) if all_likelihoods is not None else likelihoods

        visualize_scatterplots(
            scores = np.stack([all_dimensionality_scores, all_likelihoods]).T,
            column_names=["dimensionality", "likelihood"],
        )
        