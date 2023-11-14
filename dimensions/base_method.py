
from torch.utils.data import DataLoader
import typing as th
import numpy as np
from model_zoo import Writer

class BaseIntrinsicDimensionEstimationMethod:
    """
    This is a base module for estimating local intrinsic dimension.
    
    This method can use the train_dataloader and valid_dataloader to perform any
    kind of preprocessing or training required for LID estimation.
    
    Finally, the LID estimation is done on the test_dataloader.
    
    writer, is the logger used for this module.
    """
    def __init__(
        self,
        train_dataloader: th.Optional[DataLoader] = None,
        valid_dataloader: th.Optional[DataLoader] = None,
        test_dataloader: th.Optional[DataLoader] = None,
        writer: th.Optional[Writer] = None,
        device: str = 'cpu',
        ambient_dim: th.Optional[int] = None,
    ):
        if test_dataloader is None:
            raise ValueError("With no test dataloader given, the method cannor return anything!")
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.writer = writer
        if writer is None:
            raise ValueError("Writer should be specified!")
        self.device = device
        self.ambient_dim = ambient_dim
        
    def fit(
        self,
    ):
        raise Exception("No fit function implemented for the current intrinsic dimension estimation method!")
    
    def estimate_lid(
        self, x
    ):
        """

        Args:
            x : A datapoint in any format the loaders are in
        Returns:
            A numpy array of the LID estimates for all the datapoints.
        """
        raise Exception("No estimation implemented for the current method!")
    
    def estimate_id(
        self,
    ):
        """
        By default, it averages over all the datapoints in the test loader
        and returns the average LID estimates as the output. However,
        in many cases where an algorithm for global ID estimation is available,
        one can override the function directly.
        """
        avg_weighted = 0.0
        sm = 0.0
        
        for x in self.test_dataloader:
            sm += x.shape[0]
            avg_weighted += x.shape[0] * np.sum(self.estimate_lid(x))
    
        return avg_weighted / sm
    
    def custom_run(
        self,
    ):
        raise NotImplementedError("Not implemented!")
    