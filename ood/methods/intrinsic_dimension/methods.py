

import torch
from ood.base_method import OODBaseMethod
from .latent_statistics import GaussianConvolutionRateStatsCalculator
import typing as th
from ood.methods.intrinsic_dimension.utils import buffer_loader
from tqdm import tqdm
from ood.base_method import OODBaseMethod
import typing as th
from ood.visualization import visualize_histogram, visualize_trends, visualize_scatterplots
import numpy as np
from tqdm import tqdm
import wandb
import dypy as dy
from .utils import buffer_loader, get_device_from_loader
from .latent_statistics import LatentStatsCalculator
from .encoding_model import EncodingModel
import torchvision

class RadiiTrend(OODBaseMethod):
    """
    This OOD detection method visualizes trends of the latent statistics that are being calculated in the ood.methods.linear_approximations.latent_statistics.
    
    You specify a latent_statistics_calculator_class and a latetn_statistics_calculator_args and it automatically instantiates a latent statistics calculator.
    
    """
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        
        
        # Latent statistics calculator
        latent_statistics_calculator_class: th.Optional[str] = None,
        latent_statistics_calculator_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        # for logging args
        verbose: int = 0,
        
        # The range of the radii to show in the trend
        radii_range: th.Optional[th.Tuple[float, float]] = None,
        radii_n: int = 100,
        
        # visualize the trend with or without the standard deviations
        with_std: bool = False,
        
    ):
        """
        Inherits from EllipsoidBaseMethod and adds the visualization of the trend.
        
        Args:
            radii_range (th.Optional[th.Tuple[float, float]], optional): The smallest and largest distance to consider.
            radii_n (int, optional): The number of Radii scales in the range to consider. Defaults to 100.
            verbose: Different levels of logging, higher numbers means more logging
            
        """
        super().__init__(
            likelihood_model = likelihood_model,
            x_loader=x_loader,
            in_distr_loader=in_distr_loader,
        )
        self.latent_statistics_calculator: LatentStatsCalculator = dy.eval(latent_statistics_calculator_class)(likelihood_model, **(latent_statistics_calculator_args or {}))
        self.verbose = verbose
        self.radii = np.linspace(*radii_range, radii_n)
        self.with_std = with_std
        
    def _get_trend(self, loader):
               
        # split the loader and calculate the trend
        trends = [] # This is the final trend
        for i, r in enumerate(self.radii):
                
            values = self.latent_statistics_calculator.calculate_statistics(
                r,
                loader=loader,
                # use the cache when the first computation is done 
                use_cache=i > 0,
            )
            trends.append(values)
        
        trends = np.stack(trends).T
        # turns into a (datacount x r_range) numpy array
        return np.stack(trends, axis=0)
        
    def run(self):
        """
        This function runs the calculate statistics function over all the different
        r values on the entire ood_loader that is taken into consideration.
        """
        trend = self._get_trend(self.x_loader)
         
        visualize_trends(
            scores=trend,
            t_values=self.radii,
            x_label="r",
            y_label=self.latent_statistics_calculator.get_label(),
            title=f"Trend of the average {self.latent_statistics_calculator.get_name()}",
            with_std=self.with_std,
        )

class IntrinsicDimensionOODDetection(OODBaseMethod):
    """
    This method computes the intrinsic dimensionality using intrinsic dimension.
    
    The intrinsic dimension estimation technique used has a measurement hyper parameters inspired by LIDL: https://arxiv.org/abs/2206.14882
    We use a certain evaluation radius r (denoted as evaluate_r).
    This method either hard sets that to a certain value and computes the intrinsic dimensionalities
    or it uses an adaptive approach for setting that.
    
    Adaptive scaling: We have a dimension_tolerance parameter (a ratio from 0.0 to 1.0) which determines what 
    portion of the ID/D should we tolerate for our training samples.
    
    For example, a ratio 'r' with a dataset with ambient dimension D, means that the scaling would be set adaptively
    in a way so that the intrinsic dimension of the training data lies somewhere in the range of [(1 - r) * D, D]
    That way, everything is relative to the scale and a lower intrinsic dimension practically means that it is lower than
    the training data it's been originally trained on.
    
    Ideally, there should be no reason to do this adaptive scaling because if the model is a perfect fit, it will fit the data
    manifold, but in an imperfect world where model fit is not as great, this is the way to go!
    
    Finally, the log-likelihood-(vs)-Intrinsic-Dimension is plotted against each other in a scatterplot. That scatterplot is then used for
    coming up with simple thresholding techniques for OOD detection.
    """
    def __init__(
        self,
        
        # The basic parameters passed to any OODBaseMethod
        likelihood_model: torch.nn.Module,    
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        
        
        # The intrinsic dimension calculator args
        intrinsic_dimension_calculator_args: th.Optional[th.Dict[str, th.Any]] = None,
        # for logging args
        verbose: int = 0,
        
        # Hyper-parameters relating to the scale parameter that is being computed
        evaluate_r: float = -20,
        adaptive_measurement: bool = False,
        percentile: int = 10,
        dimension_tolerance: float = 0.0,
        adaptive_train_data_buffer_size: int = 15,    
    ):
        """
        Initializes the IntrinsicDimensionScore class for outlier detection using intrinsic dimensionality.

        Args:
            likelihood_model (torch.nn.Module): The likelihood model for the data.
            x (torch.Tensor, optional): A single data tensor.
            x_batch (torch.Tensor, optional): Batch of data tensors.
            x_loader (torch.utils.data.DataLoader, optional): DataLoader containing the dataset.
            logger (any, optional): Logger to use for output and debugging.
            in_distr_loader (torch.utils.data.DataLoader, optional): DataLoader containing the in-distribution dataset.
            intrinsic_dimension_calculator_args (dict, optional): Dictionary containing arguments for the intrinsic dimension calculator.
            verbose (int, optional): Verbosity level for logging. Defaults to 0.
            evaluate_r (float, optional): The radius to show in the trend for evaluating intrinsic dimension. Defaults to -20.
            adaptive_measurement (bool, optional): Flag indicating whether to use adaptive scaling for the radius. Defaults to False.
            percentile (float): When adaptive measurement is set to True, this percentile alongside the dimension_tolerance adaptively pick the scale
                                used for LID estimation.
            dimension_tolerance (float): A number between 0 and 1. A dimension_tolerance of '0.1` means that the scale is picked so that (almost all of)
                                         the training data has intrinsic dimension in the range [D - 0.1 * D, D]
            adaptive_train_data_buffer_size: (float): This determines the buffer of the in distribution loader that we take into consideration.
        """
            
        super().__init__(
            x_loader=x_loader, 
            likelihood_model=likelihood_model, 
            in_distr_loader=in_distr_loader, 
        )
        
        self.latent_statistics_calculator = GaussianConvolutionRateStatsCalculator(likelihood_model, **(intrinsic_dimension_calculator_args or {}))
        
        self.verbose = verbose
        
        self.evaluate_r = evaluate_r
        
        
        # When set to True, the adaptive measure will set the scale according to the training data in a smart way
        self.adaptive_measurement = adaptive_measurement
        self.dimension_tolerance = dimension_tolerance
        self.percentile = percentile
        self.adaptive_train_data_buffer_size = adaptive_train_data_buffer_size
        
        
    def run(self):
        
        # use training data to perform adaptive scaling of fast-LIDL
        # pick a subset of the training data loader for this adaptation
        buffer = buffer_loader(self.in_distr_loader, self.adaptive_train_data_buffer_size, limit=1)
        for _ in buffer:
            inner_loader = _
        
        if self.adaptive_measurement:
            # get all the likelihoods and detect outliers of likelihood for your training data
            all_likelihoods = None
            for x in inner_loader:
                D = x.numel() // x.shape[0]

                with torch.no_grad():
                    likelihoods = self.likelihood_model.log_prob(x).cpu().numpy().flatten()
                all_likelihoods = likelihoods if all_likelihoods is None else np.concatenate([all_likelihoods, likelihoods])
            
            print("D = ", D)            
            L = -20.0 
            R = np.log(0.99)
            
            if self.verbose > 0:
                bin_search = tqdm(range(20), desc="binary search to find the scale")
            else:
                bin_search = range(20)
            
            # perform a binary search to come up with a scale so that almost all the training data has dimensionality above
            # that threshold
            for ii in bin_search:
                mid = (L + R) / 2
                dimensionalities = self.latent_statistics_calculator.calculate_statistics(
                    mid,
                    loader=inner_loader,
                    use_cache=(ii != 0),
                )
                sorted_dim = np.sort(dimensionalities)
                dim_comp = sorted_dim[int(float(self.percentile) / 100 * len(sorted_dim))]

                if dim_comp < - D * self.dimension_tolerance:
                    R = mid
                else:
                    L = mid
            self.evaluate_r = L
        
        if self.verbose > 0:
            print("running with scale:", self.evaluate_r)
        
        
        all_likelihoods = None
        all_dimensionalities = None
        all_dimensionalities_clamped = None
        
        for x in self.x_loader:
            D = x.numel() // x.shape[0]
            with torch.no_grad():
                likelihoods = self.likelihood_model.log_prob(x).cpu().numpy().flatten()
                all_likelihoods = np.concatenate([all_likelihoods, likelihoods]) if all_likelihoods is not None else likelihoods
                
        inner_dimensionalities  = self.latent_statistics_calculator.calculate_statistics(
            self.evaluate_r,
            loader=self.x_loader,
            use_cache=False,
        ) 
        
        inner_dimensionalities_clamped = np.where(
            inner_dimensionalities + self.dimension_tolerance * D > 0, 
            np.zeros_like(inner_dimensionalities), 
            inner_dimensionalities + self.dimension_tolerance * D
        )
            
        all_dimensionalities = np.concatenate([all_dimensionalities, inner_dimensionalities]) if all_dimensionalities is not None else inner_dimensionalities
        all_dimensionalities_clamped = np.concatenate([all_dimensionalities_clamped, inner_dimensionalities_clamped]) if all_dimensionalities_clamped is not None else inner_dimensionalities_clamped
        
        visualize_scatterplots(
            scores = np.stack([all_likelihoods, all_dimensionalities]).T,
            column_names=["log-likelihood", "relative-ID-(d-D)"],
        )
        
        single_scores = all_likelihoods * np.exp(2 * self.evaluate_r) + all_dimensionalities_clamped
       
        visualize_histogram(
            scores=single_scores,
            x_label="heuristic_score",
        )

class ReconstructionCheck(OODBaseMethod):
    """
    This function performs a reconstruction check by having an encoding model within.
    What it does is that for x_loader it performs and encoding following by a decoding
    and visualizes the first pictures alongside the reconstructed ones.
    For flow models, this is a good check for model invertibility.
    For VAEs, it shows how well the model reconstructs.
    """
    def __init__(
        self,
        
        # The basic parameters passed to any OODBaseMethod
        likelihood_model: torch.nn.Module,    
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        
        
        # The intrinsic dimension calculator args
        encoding_model_class: th.Optional[str] = None,
        encoding_model_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        # Hyper-parameters relating to the scale parameter that is being computed
        visualization_nrow: int = 4, 
    ):
        """
        Initializes the IntrinsicDimensionScore class for outlier detection using intrinsic dimensionality.

        Args:
            likelihood_model (torch.nn.Module): The likelihood model for the data.
            x (torch.Tensor, optional): A single data tensor.
            x_batch (torch.Tensor, optional): Batch of data tensors.
            x_loader (torch.utils.data.DataLoader, optional): DataLoader containing the dataset.
            logger (any, optional): Logger to use for output and debugging.
            in_distr_loader (torch.utils.data.DataLoader, optional): DataLoader containing the in-distribution dataset.
            encoding_model_class: (str) this shows the class that is used for encoding and decoding
            visualization_nrow (int) the number of rows and columns to visualize the pictures on.
        """
            
        super().__init__(
            x_loader=x_loader, 
            likelihood_model=likelihood_model, 
            in_distr_loader=in_distr_loader, 
        )
        
        self.encoding_model: EncodingModel = dy.eval(encoding_model_class)(likelihood_model, **encoding_model_args)
        
        self.visualization_nrow = visualization_nrow
        
        
    def run(self):
        original = []
        reconstructed = []
        device = get_device_from_loader(self.x_loader)
        
        rem = self.visualization_nrow ** 2
        for x_batch in self.x_loader:
            rem -= len(x_batch)
            x_batch = x_batch.to(device)
            if not self.encoding_model.diff_transform:
                x_batch = self.likelihood_model._data_transform(x_batch)
            x_reconstructed = self.encoding_model.decode(self.encoding_model.encode(x_batch))
            if not self.encoding_model.diff_transform:
                x_reconstructed = self.likelihood_model._inverse_data_transform(x_reconstructed)
            original.append(x_batch.cpu())
            reconstructed.append(x_reconstructed.cpu())
            if rem <= 0:
                break
        
        original = torch.cat(original)
        reconstructed = torch.cat(reconstructed)
        
        if len(original) > self.visualization_nrow ** 2:
            original = original[:self.visualization_nrow ** 2]
            reconstructed = reconstructed[:self.visualization_nrow ** 2]
        
        original = torchvision.utils.make_grid(original, nrow=self.visualization_nrow)
        reconstructed = torchvision.utils.make_grid(reconstructed, nrow=self.visualization_nrow)
        
        wandb.log({
            "invertibility/original": [wandb.Image(original, caption="original images")]
        })
        wandb.log({
            "invertibility/reconstructed": [wandb.Image(reconstructed, caption="reconstructed images")]
        })