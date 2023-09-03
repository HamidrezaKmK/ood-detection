# create an abstract class named OODBaseMethod
# with an abstract method named run
from abc import ABC, abstractmethod
import typing as th
import torch


class OODBaseMethod(ABC):
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        x: th.Optional[torch.Tensor] = None,
        x_batch: th.Optional[torch.Tensor] = None,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        logger: th.Optional[th.Any] = None,
        progress_bar: bool = False,
    ) -> None:
        super().__init__()
        if x is None and x_batch is None and x_loader is None:
            raise ValueError("Either x or x_batch or x_loader must be provided!")
        self.x = x
        self.x_batch = x_batch
        self.likelihood_model = likelihood_model
        self.logger = logger
        self.progress_bar = progress_bar
        self.x_loader = x_loader
        self.in_distr_loader = in_distr_loader

        
        if self.x is not None:    
            self.x_batch = self.x.unsqueeze(0)
        
        if self.x_batch is not None:
            # create a loader with that single batch
            self.x_loader = [self.x_batch]
        

    @abstractmethod
    def run(self):
        raise NotImplementedError("run method not implemented!")
