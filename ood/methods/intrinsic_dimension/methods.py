
import numpy as np
import torch
from scipy import stats
from ood.methods.base_method import OODBaseMethod
import typing as th
from ood.methods.linear_approximations import GaussianConvolutionRateStatsCalculator
from ood.methods.utils import buffer_loader
from ood.visualization import visualize_scatterplots, visualize_histogram
from tqdm import tqdm
import wandb
from sklearn.covariance import EllipticEnvelope


def identify_threshold(
    likelihoods,
    dimensionalities,
    type: th.Literal['simple-mean', 'simple-percentile', 'robust-covariance', 'zscore', 'iqr'] = 'simple-percentile',
    iqr_coeff: float = 1.5, 
    z_score_threshold: float = 2,
    contamination: float = 0.1,
    percentile: int = 10,
):
    # NOTE: This should be a very non-restrictive outlier detection method! Playing around with 
    # the hyperparameters might cause it to break apart.
    
    if type == 'simple-mean':
        return np.mean(dimensionalities)
    elif type == 'simple-percentile':
        sorted_dim = np.sort(dimensionalities)
        return sorted_dim[int(float(percentile) / 100 * len(sorted_dim))]
    elif type == 'iqr':
        # Calculate percentiles
        Q1 = np.percentile(likelihoods, 25)
        Q3 = np.percentile(likelihoods, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_coeff * IQR
        upper_bound = Q3 + iqr_coeff * IQR
        x_msk = (lower_bound <= likelihoods) & (likelihoods <= upper_bound)
        # Calculate percentiles
        Q1 = np.percentile(dimensionalities, 25)
        Q3 = np.percentile(dimensionalities, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_coeff * IQR
        upper_bound = Q3 + iqr_coeff * IQR
        y_msk = (lower_bound <= dimensionalities) & (dimensionalities <= upper_bound)
        
        return np.min(dimensionalities[x_msk & y_msk])
    elif type == 'zscore':
        x_z_scores = np.abs(stats.zscore(likelihoods))
        y_z_scores = np.abs(stats.zscore(dimensionalities))
        return np.min(dimensionalities[(x_z_scores <= z_score_threshold) & (y_z_scores <= z_score_threshold)])
    elif type == 'robust-covariance':
        clf = EllipticEnvelope(contamination=contamination, random_state=0)
        dat = np.stack([likelihoods, dimensionalities]).T
        pred = clf.fit(dat).predict(dat)
        return np.min(dimensionalities[np.where(pred < 0, np.zeros_like(pred), np.ones_like(pred))])
    else:
        raise NotImplementedError(f"The inlier detection algorithm {type} is non-existant!")
    
class IntrinsicDimensionScore(OODBaseMethod):
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
        x: th.Optional[torch.Tensor] = None,
        x_batch: th.Optional[torch.Tensor] = None,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        logger: th.Optional[th.Any] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        
        
        # The intrinsic dimension calculator args
        intrinsic_dimension_calculator_args: th.Optional[th.Dict[str, th.Any]] = None,
        # for logging args
        verbose: int = 0,
        
        # The range of the radii to show in the trend
        evaluate_r: float = -20,
        adaptive_measurement: bool = False,
        # visualization arguments
        visualization_args: th.Optional[th.Dict[str, th.Any]] = None,

        # in distr training
        ood_loader_limit: th.Optional[int] = None,
        ood_loader_buffer_size: int = 60,
                
        # in_distribution training hyperparameters
        in_distr_loader_buffer_size: int = 10,
        
        dimension_tolerance: float = 0.33,
        
        thresholding_args: th.Optional[th.Dict[str, th.Any]] = None
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
            visualization_args (dict, optional): Dictionary containing arguments for visualization.
            ood_loader_limit (int, optional): Limit on the number of samples from the OOD data loader.
            ood_loader_buffer_size (int, optional): Buffer size for the OOD data loader. Defaults to 60.
            in_distr_loader_buffer_size (int, optional): Buffer size for the in-distribution data loader. Defaults to 5.
            dimension_tolerance (float, optional): Tolerance parameter for dimension scaling, specified as a ratio between 0.0 and 1.0. Defaults to 0.33.

        """
            
        super().__init__(
            x_loader=x_loader, 
            x=x, 
            x_batch=x_batch, 
            likelihood_model=likelihood_model, 
            logger=logger, 
            in_distr_loader=in_distr_loader, 
        )
        
        self.latent_statistics_calculator = GaussianConvolutionRateStatsCalculator(likelihood_model, **(intrinsic_dimension_calculator_args or {}))
        
        self.verbose = verbose
        
        self.evaluate_r = evaluate_r
        
        self.visualization_args = visualization_args or {}
        
        # The number of datapoints from training to consider for tuning the scale of LIDL
        self.in_distr_loader_buffer_size = in_distr_loader_buffer_size
        
        self.ood_loader_limit = ood_loader_limit
        self.ood_loader_buffer_size = ood_loader_buffer_size
        
        # When set to True, the adaptive measure will set the scale according to the training data in a smart way
        self.adaptive_measurement = adaptive_measurement
        
        self.dimension_tolerance = dimension_tolerance
        
        self.thresholding_args = thresholding_args or {}
        
    def run(self):
        # (1) compute the average dimensionality score of the in-distribution data
        calculate_statistics_additional_args = {}
        
        # use training data to perform adaptive scaling of fast-LIDL
        if self.adaptive_measurement:
            buffer = buffer_loader(self.in_distr_loader, self.in_distr_loader_buffer_size, limit=1)
            for inner_loader in buffer:
                # get all the likelihoods and detect outliers of likelihood for your training data
                all_likelihoods = None
                for x in inner_loader:
                    D = x.numel() // x.shape[0]
                    with torch.no_grad():
                        likelihoods = self.likelihood_model.log_prob(x).cpu().numpy().flatten()
                    all_likelihoods = likelihoods if all_likelihoods is None else np.concatenate([all_likelihoods, likelihoods])
                
                L = -20.0 
                R = np.log(0.99)
                
                if self.verbose > 0:
                    rng = tqdm(range(20), desc="binary search to find the scale")
                else:
                    rng = range(20)
                
                # perform a binary search to come up with a scale so that almost all the training data has dimensionality above
                # that threshold
                for ii in rng:
                    mid = (L + R) / 2
                    dimensionalities = self.latent_statistics_calculator.calculate_statistics(
                        mid,
                        loader=inner_loader,
                        use_cache=(ii != 0),
                        log_scale=True,
                        order=1,
                    )
                    
                    if identify_threshold(all_likelihoods, dimensionalities, **self.thresholding_args) < - D * self.dimension_tolerance:
                        R = mid
                    else:
                        L = mid
                self.evaluate_r = L
        
        print("running with scale:", self.evaluate_r)
        
        # (2) compute the average dimensionality score of the OOD data
        if self.verbose > 0:
            buffer = tqdm(
                buffer_loader(self.x_loader, self.ood_loader_buffer_size, limit=self.ood_loader_limit),
                desc="Calculating OOD dimensionalities for loader in buffer",
                total=self.ood_loader_limit,
            )
        else:
            buffer = buffer_loader(self.x_loader, self.ood_loader_buffer_size, limit=self.ood_loader_limit)
        
        all_likelihoods = None
        all_dimensionalities = None
        all_dimensionalities_clamped = None
        
        for inner_loader in buffer:
            for x in inner_loader:
                D = x.numel() // x.shape[0]
                with torch.no_grad():
                    likelihoods = self.likelihood_model.log_prob(x).cpu().numpy().flatten()
                    all_likelihoods = np.concatenate([all_likelihoods, likelihoods]) if all_likelihoods is not None else likelihoods
                    
            inner_dimensionalities  = self.latent_statistics_calculator.calculate_statistics(
                self.evaluate_r,
                loader=inner_loader,
                use_cache=False,
                log_scale=True,
                order=1,
                **calculate_statistics_additional_args,
            ) 
            inner_dimensionalities += self.dimension_tolerance * D
            
            inner_dimensionalities_clamped = np.where(inner_dimensionalities > 0, np.zeros_like(inner_dimensionalities), inner_dimensionalities)
                
            all_dimensionalities = np.concatenate([all_dimensionalities, inner_dimensionalities]) if all_dimensionalities is not None else inner_dimensionalities
            all_dimensionalities_clamped = np.concatenate([all_dimensionalities_clamped, inner_dimensionalities_clamped]) if all_dimensionalities_clamped is not None else inner_dimensionalities_clamped
        
        visualize_scatterplots(
            scores = np.stack([all_likelihoods, all_dimensionalities]).T,
            column_names=["log-likelihood", "relative-ID-d-D"],
        )
        
        single_scores = all_likelihoods * np.exp(2 * self.evaluate_r) + all_dimensionalities_clamped
        table = wandb.Table(columns=['scores'], data=[[x] for x in single_scores])
    
        wandb.log({"scores_histogram": wandb.plot.histogram(table, "scores", title="computed empirical OOD scores")})