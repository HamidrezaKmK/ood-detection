"""
This file contains wrappers around deep generative models, with functions for encode and decode
that can be used for ood detection.
"""
import torch
from tqdm import tqdm
import typing as th
from .utils import stack_back_iterables

class EncodingModel:
    """
    This is a class that is used for likelihood models that have an encode and
    decode functionality with them. Here, the encode and decode function are 
    abstract. For example, for a flow model, that would be running the transformation.
    And for a VAE model, that would be running the encoder and the decoder.

    Moreover, this class has a set of knobs designed for optimizing the computation
    of the Jacobian.
    """
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        # These are computation-related knobs
        # For example, with use_functorch and use_forward_mode
        # you can specify how to compute the Jacobian
        # With computation_batch_size you determine how large your jacobian batch
        # is going to be
        use_vmap: bool = False,
        use_functorch: bool = True,
        use_forward_mode: bool = True,
        computation_batch_size: th.Optional[int] = None,
        # verbosity
        verbose: int = 0,
    ) -> None:
        
        self.likelihood_model = likelihood_model
        
        # Set the model to evaluation mode if its not already!
        # This is because we need the model to act in a deterministic way
        self.likelihood_model.eval()
        
        self.use_functorch = use_functorch
        self.use_forward_mode = use_forward_mode
        self.use_vmap = use_vmap
        self.computation_batch_size = computation_batch_size
        
        self.verbose = verbose
    
    def decode(self, z):
        """Decodes the input from a standard Gaussian latent space 'z' onto the data space."""
        raise NotImplementedError("You must implement a decode method for your model.")
    
    def encode(self, x):
        """Encodes the input 'x' onto a standard Gaussian Latent Space."""
        raise NotImplementedError("You must implement an encode method for your model.")
    
    def calculate_jacobian(self, loader, stack_back: bool = True):
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
        """
    
        
        if self.verbose > 1:
            loader_decorated = tqdm(loader)
        else:
            loader_decorated = loader
        
        jax = []
        z_values = []
           
        for x_batch in loader_decorated:
            
            # encode to obtain the latent representation in the Gaussian space
            z = self.encode(x_batch)
            
            # Since Jacobian computation is heavier than the normal batch_wise
            # computations, we have another level of batching here
            step = self.computation_batch_size
            if self.computation_batch_size is None:
                step = z.shape[0]
            
            progress = 0
            for l in range(0, z.shape[0], step):
                progress += 1
                if self.verbose > 1:
                    loader_decorated.set_description(f"Computing latent jacobians [{progress}/{z.shape[0] // step}]")
        
                r = min(l + step, z.shape[0])
                
                # Get a batch of latent representations
                z_s = z[l:r]

                # Calculate the jacobian of the decode function
                if self.use_functorch:
                    jac_fn = torch.func.jacfwd if self.use_forward_mode else torch.func.jacrev
                    if self.use_vmap:
                        # optimized implementation with vmap, however, it does not work as of yet
                        jac_until_now = torch.func.vmap(jac_fn(self.decode))(z_s)
                    else:
                        jac_until_now = jac_fn(self.decode)(z_s)
                else:
                    jac_until_now = torch.autograd.functional.jacobian(self.decode, z_s)

                # Reshaping the jacobian to be of the shape (batch_size, latent_dim, latent_dim)
                if self.use_vmap:
                    jac = jac_until_now.reshape(z_s.shape[0], -1, z_s.numel() // z_s.shape[0])
                else:
                    jac_until_now = jac_until_now.reshape(z_s.shape[0], -1, z_s.shape[0], z_s.numel() // z_s.shape[0])
                    jac = []
                    for j in range(jac_until_now.shape[0]):
                        jac.append(jac_until_now[j, :, j, :])
                    jac = torch.stack(jac)
                
                z_values.append(z_s.cpu().detach())
                jax.append(jac.cpu().detach())
                        
        return stack_back_iterables(loader, jax, z_values) if stack_back else jax, z_values


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
    def encode(self, x):
        # turn off the gradient for faster computation
        # because we don't need to change the model parameters
        # self.likelihood_model.eval()
        with torch.no_grad():
            try:
                z = self.likelihood_model._nflow.transform_to_noise(x)
            except ValueError as e:
                z = self.likelihood_model._nflow.transform_to_noise(x.unsqueeze(0))
                z = z.squeeze(0)
            return z

    def decode(self, z):
        # turn off the gradient for faster computation
        # because we don't need to change the model parameters
        # self.likelihood_model.eval()
        with torch.no_grad():
            try:
                x, logdets = self.likelihood_model._nflow._transform.inverse(z)
            except ValueError as e:
                x, logdets = self.likelihood_model._nflow._transform.inverse(z.unsqueeze(0))
                x = x.squeeze(0)
            return x
