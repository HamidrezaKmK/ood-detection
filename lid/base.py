from abc import ABC, abstractmethod

import torch
from einops import repeat
import numpy as np

class LocalIntrinsicDimensionEstimator(ABC):
    
    def __init__(
        self,
        data,
        ground_truth_lid = None,
    ):
        """
        Initialize the estimator with the data that we are planning to use for LID estimation 
        and also take in an optional ground truth LID if it is available
        """
        self.data = data
    
    @abstractmethod
    def fit(
        self,
    ):
        '''Fit the estimator to the data'''
        pass
    
    @abstractmethod
    def estimate_lid(
        self,
        x: Union[torch.Tensor, np.ndarray],
        scale: Optional[Union[float, int]] = None,
    ):
        """
        Estimate the local intrinsic dimension of the data at given points. 
        The input is batched, so the output should be batched as well.
        there is also a scale parameter that can be used to estimate the LID at different scales
        by scale we mean what level of data perturbation do we ignore. 
        As an extreme example, for an excessively large scale, the LID will be the same for all points and equal to 0.
        On the other hand, for an excessively small scale, the LID will be the same for all points and equal to the ambient dimension
        
        Args:
            x (torch.Tensor or np.ndarray):
                The point at which to estimate the LID
            scale (Optional[Union[float, int]]):
                The scale at which to estimate the LID. If None, the scale will be estimated from the data
        Returns:
            lid (torch.Tensor or np.ndarray):
                The estimated LID at all the given points in the same order, it should be the same type as the input x
        """
        pass
    
    @abstractmethod
    def estimate_id(
        self,
    ):
        '''Estimate the intrinsic dimension of the data if there is a global intrinsic dimensionality associated with the data'''
        pass

class ModelBasedLID(LocalIntrinsicDimensionEstimator):
    '''An abstract class for estimators that use a model to estimate LID'''
    def __init__(
        self,
        data: torch.utils.data.Dataset,
        model: torch.nn.Module,
        train_fn: Callable[[torch.nn.Module, torch.utils.data.Dataset], torch.nn.Module] = None,
    ):
        """

        Args:
            data (torch.nn.Module):
                The dataset used for LID estimation and training
            model (torch.nn.Module):
                The likelihood model that the LID estimator will use for estimation
            train_fn (Callable[[torch.nn.Module, torch.utils.data.Dataset], torch.nn.Module]): 
                This is a function that takes in a torch module and a torch dataset and trains the module on the dataset.
        """
        super().__init__(data)
        self.model = model
        self.train_fn = train_fn if train_fn is not None else (lambda model, data: model)
        
    def fit(self):
        '''Fit the estimator to the data'''
        self.model = self.train_fn(self.model, self.data)
        
        