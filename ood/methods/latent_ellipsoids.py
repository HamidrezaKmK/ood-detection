from .base_method import OODBaseMethod
import typing as th
import torch

class LatentEllipsoids(OODBaseMethod):
    """
    This class of methods works only on Gaussian latent spaces, for example, VAEs and Normflows.
    One might potentially extend these to general models via an extra level of transformation. However,
    for the sake of simplicity, we will only consider this to work on flow models.
    """
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        # 
        x: th.Optional[torch.Tensor] = None,
        x_batch: th.Optional[torch.Tensor] = None,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        logger: th.Optional[th.Any] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        #
        use_functorch: bool = True,
        use_forward_mode: bool = True,
        computation_batch_size: int = 1,
        #
        predefined_r: th.Optional[float] = None,
    ):
        super().__init__(
            x_loader=x_loader, 
            x=x, 
            x_batch=x_batch, 
            likelihood_model=likelihood_model, 
            logger=logger, 
            in_distr_loader=in_distr_loader, 
        )
        
        self.use_functorch = use_functorch
        self.use_forward_mode = use_forward_mode
        self.computation_batch_size = computation_batch_size
        
        self.predefined_r = predefined_r
        
        if self.x is not None:
            self.x_batch = self.x.unsqueeze(0)
        
        if self.x_batch is not None:
            # create a loader with that single batch
            self.x_loader = [(self.x_batch, None, None)]
    
    def decode(self, z):
        """Decodes the input from a standard Gaussian latent space 'z' onto the data space."""
        raise NotImplementedError("You must implement a decode method for your model.")
    
    def encode(self, x):
        """Encodes the input 'x' onto a standard Gaussian Latent Space."""
        raise NotImplementedError("You must implement an encode method for your model.")
    
    def calculate_reference(self):
        pass
    
    def run(self):
        for x_batch, _, _ in self.x_loader:
            pass