"""
This file contains wrappers around deep generative models, with functions for encode and decode
that can be used for ood detection.
"""
import torch
from tqdm import tqdm
import typing as th
from ood.methods.utils import stack_back_iterables
import numpy as np
import functools
import wandb

class EncodingModel:
    """
    This is a class that is used for likelihood models that have an encode and
    decode functionality with them. Here, the encode and decode function are 
    abstract. For example, for a flow model, that would be running the transformation.
    And for a VAE model, that would be running the encoder and the decoder.

    Moreover, this class has a set of knobs designed for optimizing the computation
    of the Jacobian and heavily uses functorch functionalities if specified.
    """
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        use_vmap: bool = False,
        use_functorch: bool = True,
        use_forward_mode: bool = True,
        chunk_size: th.Optional[int] = None,
        # verbosity
        verbose: int = 0,
        #
        diff_transform: bool = False,
    ) -> None:
        """
        likelihood_model:   The likelihood model that is used for the encoding and decoding, it is of type
                            torch.nn.Module
        use_vmap:           Whether to use the vmap function of functorch or not.
        use_functorch:      Whether to use functorch or not. If set to False, then the torch.autograd.functional
        use_forward_mode:   Whether to use the forward mode or the reverse mode of functorch.
        chunk_size:         The size of the chunks to be used for the computation of the Jacobian. If set to None,
                            then the whole batch is used for the computation of the Jacobian.
        verbose:            The verbosity level of the computation. If set to 0, then no progress bar is shown.  
                            if verbose > 0, then a progress bar is shown for the computation of the Jacobian. 
        """
        
        self.likelihood_model = likelihood_model
        
        # Set the model to evaluation mode if its not already!
        # This is because we need the model to act in a deterministic way
        self.likelihood_model.eval()
        # iterate over all the parameters of likelihood_model and turn off their gradients
        # this is because we assume that we don't want to change the parameters of the model
        # and do any training anymore
        for param in self.likelihood_model.parameters():
            param.requires_grad = False
            
        self.use_functorch = use_functorch
        self.use_forward_mode = use_forward_mode
        self.use_vmap = use_vmap
        self.chunk_size = chunk_size
        
        self.verbose = verbose
        
        self.diff_transform = diff_transform
    
    def decode(self, z):
        """Decodes the input from a standard Gaussian latent space 'z' onto the data space."""
        raise NotImplementedError("You must implement a decode method for your model.")
    
    def encode(self, x):
        """Encodes the input 'x' onto a standard Gaussian Latent Space."""
        raise NotImplementedError("You must implement an encode method for your model.")
    
    def calculate_jacobian(
        self, 
        loader, 
        flatten: bool = True
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
        
        This function calculates the encoding of each of the datapoints,
        then using the latent representation, calculates the Jacobian of
        the decoded function. This is done batch-wise to optimize the computation.
        
        Args:
            loader:     The loader of the data points.
        """
    
        # set a progress bar if verbose > 0
        if self.verbose > 0:
            loader_decorated = tqdm(loader)
        else:
            loader_decorated = loader
        
        jax = []
        z_values = []
           
        for x_batch in loader_decorated:
            
            if not self.diff_transform:
                x_batch = self.likelihood_model._data_transform(x_batch)
                
            # encode to obtain the latent representation in the Gaussian space
            
            z = self.encode(x_batch)
            # count the number of NaNs in z
                
            # Since Jacobian computation is heavier than the normal batch_wise
            # computations, we have another level of batching here
            step = self.chunk_size
            if self.chunk_size is None:
                step = z.shape[0]
            
            progress = 0
            half_len = None
            
            for l in range(0, z.shape[0], step):
                progress += 1
                if self.verbose > 1:
                    loader_decorated.set_description(f"Computing jacobian chunk [{progress}/{z.shape[0] // step}]")
        
                r = min(l + step, z.shape[0])
                
                # Get a batch of latent representations
                z_s = z[l:r]
                
                # check if z_s contains any NaN values
                if torch.isnan(z_s).any():
                    print(">> NaN values detected in the latent representation. Skipping this batch.")
                    continue
                
                # Calculate the jacobian of the decode function
                if self.use_functorch:
                    jac_fn = torch.func.jacfwd if self.use_forward_mode else torch.func.jacrev
                    if self.use_vmap:
                        # optimized implementation with vmap, however, it does not work as of yet
                        jac_until_now = torch.func.vmap(jac_fn(functools.partial(self.decode, batchwise=False)))(z_s)
                    else:
                        jac_until_now = jac_fn(self.decode)(z_s)
                else:
                    jac_until_now = torch.autograd.functional.jacobian(self.decode, z_s)

                
                # Reshaping the jacobian to be of the shape (batch_size, latent_dim, latent_dim)
                if self.use_vmap and self.use_functorch:
                    if flatten:
                        jac = jac_until_now.reshape(z_s.shape[0], -1, z_s.numel() // z_s.shape[0])
                    else:
                        jac = jac_until_now
                else:
                    if flatten:
                        jac_until_now = jac_until_now.reshape(z_s.shape[0], -1, z_s.shape[0], z_s.numel() // z_s.shape[0])
                    jac = []
                    
                    if half_len is None:
                        half_len = 0
                        while np.prod(jac_until_now.shape[:half_len]) != np.prod(jac_until_now.shape[half_len:]):
                            half_len += 1
                    
                    for j in range(jac_until_now.shape[0]):
                        slice_indices = [j] + [slice(None)] * (half_len - 1) + [j] + [slice(None)] * (len(jac_until_now.shape) - half_len - 1)
                        jac.append(jac_until_now[tuple(slice_indices)])
                    jac = torch.stack(jac)
                if flatten:
                    z_s = z_s.reshape(z_s.shape[0], -1)
                z_values.append(z_s.cpu().detach())
                jax.append(jac.cpu().detach())

        
        # return jax, z_values 
        return stack_back_iterables(loader, jax, z_values)
        

# TODO: Implement the VAE models here                
class EncodingVAE(EncodingModel):
    """
    The EncodingModel implemented for the VAEs in the model_zoo repository.
    """
    pass

class EncodingFlow(EncodingModel):
    """
    The EncodingModel implemented for the NormFlows in the model_zoo repository.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.error_count = 0
        
    def encode(self, x, batchwise: bool = True):
        # turn off the gradient for faster computation
        # because we don't need to change the model parameters
        # self.likelihood_model.eval()
        # with torch.no_grad():
        if self.diff_transform:
            x = self.likelihood_model._data_transform(x)
        if batchwise:
            z = self.likelihood_model._nflow.transform_to_noise(x)
        else:
            z = self.likelihood_model._nflow.transform_to_noise(x.unsqueeze(0)).squeeze(0)
        return z

    def decode(self, z, batchwise: bool = True):
        # turn off the gradient for faster computation
        # because we don't need to change the model parameters
        # self.likelihood_model.eval()
        # with torch.no_grad():
        
        if batchwise:
            x, logdets = self.likelihood_model._nflow._transform.inverse(z)
        else:
            x, logdets = self.likelihood_model._nflow._transform.inverse(z.unsqueeze(0))
            x = x.squeeze(0)
        
        if self.diff_transform:
            x = self.likelihood_model._inverse_data_transform(x)

        return x
