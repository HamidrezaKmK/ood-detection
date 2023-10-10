# create an abstract class named OODBaseMethod
# with an abstract method named run
from abc import ABC, abstractmethod
import typing as th
import torch


class OODBaseMethod(ABC):
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
    ) -> None:
        super().__init__()
        self.likelihood_model = likelihood_model
        self.x_loader = x_loader
        self.in_distr_loader = in_distr_loader
        
    @abstractmethod
    def run(self):
        raise NotImplementedError("run method not implemented!")
