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
        logger: th.Optional[th.Any] = None,
    ) -> None:
        super().__init__()
        if x is None and x_batch is None:
            raise ValueError("Either x or x_batch must be provided!")
        self.x = x
        self.x_batch = x_batch
        self.likelihood_model = likelihood_model
        self.logger = logger

    @abstractmethod
    def run(self):
        raise NotImplementedError("run method not implemented!")
