from ..base_method import OODBaseMethod
import typing as th
import torch
import math
from ...visualization import visualize_histogram, visualize_trends, visualize_scatterplots
import numpy as np
from chi2comb import ChiSquared, chi2comb_cdf
from tqdm import tqdm
import wandb
import dypy as dy
from scipy.stats import norm
from scipy.special import logsumexp
from .utils import buffer_loader
from .latent_statistics import LatentStatsCalculator
import torchvision
from .latent_statistics import GaussianConvolutionStatsCalculator, GaussianConvolutionRateStatsCalculator
from math import log

class LatentBaseMethod(OODBaseMethod):
    """
    This class encompasses all of the methods that require a latent_statistics_calculator 
    to operate and depend on some linear approximation of the generative model at hand. 
    
    This includes:
    
    1.  The models that approximate the ambient ball measures 
        using latent ellipsoids in the Gaussian latent space.
        In turn, these methods might create a trend, or they might
        just calculate a score based on that particular latent statistic.
    
    2.  The models that approximate the convolution with a Gaussian with a
        pre-specified standard deviation.
        
    The main theme in common with these methods is that due to the linear approximation,
    these methods can calculate some data-space statistics in the latent space.
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
        
        # statistics calculator
        latent_statistics_calculator_class: th.Optional[str] = None,
        latent_statistics_calculator_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        # for logging args
        verbose: int = 0,
    ):
        """
        Args:
            likelihood_model (torch.nn.Module): The likelihood model with initialized parameters that work best for generation.
            x (th.Optional[torch.Tensor], optional): If the OOD detection task is only aimed for single point, this is the point.
            x_batch (th.Optional[torch.Tensor], optional): If the OOD detection task is aimed for a single batch.
            x_loader (th.Optional[torch.utils.data.DataLoader], optional): If the OOD detection methods is to be applied on the entire
                dataloader.
            logger (th.Optional[th.Any], optional): This is the logger being used.
            in_distr_loader (th.Optional[torch.utils.data.DataLoader], optional): This is the dataloader pertaining to the training data,
                in OOD detection, we have access to the training data itself and our method might do some relevant computations on that.
            verbose (int, optional): This controls the logging in our model:   
                (0) No logging
                (1) Partial logging
                (2) Many logs
            latent_statistics_calculator_class: (str) The class of the latent statistics calculator
            latent_statistics_calculator_args (dict) The arguments used for visualizing the latent statistics calculator
        """
        super().__init__(
            x_loader=x_loader, 
            x=x, 
            x_batch=x_batch, 
            likelihood_model=likelihood_model, 
            logger=logger, 
            in_distr_loader=in_distr_loader, 
        )
        
        latent_statistics_calculator_args = latent_statistics_calculator_args or {}
        if 'verbose' not in latent_statistics_calculator_args:
            latent_statistics_calculator_args['verbose'] = verbose
            
        self.latent_statistics_calculator: LatentStatsCalculator = dy.eval(latent_statistics_calculator_class)(likelihood_model, **(latent_statistics_calculator_args or {}))
        
        if self.x is not None:    
            self.x_batch = self.x.unsqueeze(0)
        
        if self.x_batch is not None:
            # create a loader with that single batch
            self.x_loader = [(self.x_batch, None, None)]
        
        self.verbose = verbose

class IntrinsicDimensionScore(LatentBaseMethod):
    """
    This class computes a continuous dimensionality metric using the derivative of the 
    Gaussian convolution and computes it at a specific 'r' value which is defined in `evaluate_r`.
    
    The derivative is computed using the analytical approach and inspired by the LIDL paper.
    
    now that dimensionality is "compared" to the in_distr_loader dimensionality and a score is computed.
    
    This score is plotted against the likelihood scores in a scatterplot.
    """
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        
        x: th.Optional[torch.Tensor] = None,
        x_batch: th.Optional[torch.Tensor] = None,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        logger: th.Optional[th.Any] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        
        
        # Latent statistics calculator
        latent_statistics_calculator_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        # for logging args
        verbose: int = 0,
        
        # The range of the radii to show in the trend
        evaluate_r: float = 1e-6,
        adaptive_measurement: bool = False,
        log_scale: bool = False,
        
        # visualization arguments
        visualization_args: th.Optional[th.Dict[str, th.Any]] = None,

        # in distr training
        ood_loader_limit: th.Optional[int] = None,
        ood_loader_buffer_size: int = 60,
                
        # in_distribution training hyperparameters
        perform_training: bool = False,
        jacobian_relative_scaling_factor: float = 1,
        automatic_jacobian_scaling: bool = False,        
        in_distr_loader_buffer_size: int = 60,

        
    ):
        self.latent_statistics_calculator: GaussianConvolutionRateStatsCalculator
        
        super().__init__(
            likelihood_model=likelihood_model,
            x=x,
            x_batch=x_batch,
            x_loader=x_loader,
            logger=logger,
            in_distr_loader=in_distr_loader,
            latent_statistics_calculator_class='ood.methods.linear_approximations.GaussianConvolutionRateStatsCalculator',
            latent_statistics_calculator_args=latent_statistics_calculator_args,
            verbose=verbose,
        )
        self.perform_training = perform_training
        
        self.evaluate_r = evaluate_r
        self.log_scale = log_scale
        
        self.visualization_args = visualization_args or {}
        
        self.in_distr_loader_buffer_size = in_distr_loader_buffer_size
        
        self.ood_loader_limit = ood_loader_limit
        self.ood_loader_buffer_size = ood_loader_buffer_size
        
        self.jacobian_relative_scaling_factor = jacobian_relative_scaling_factor
        self.automatic_jacobian_scaling = automatic_jacobian_scaling
        self.adaptive_measurement = adaptive_measurement
    
    def run(self):
        # (1) compute the average dimensionality score of the in-distribution data
        
        
        calculate_statistics_additional_args = {}
        
        if self.perform_training or self.adaptive_measurement:
            buffer = buffer_loader(self.in_distr_loader, self.in_distr_loader_buffer_size, limit=1)
            all_jtj_eigvals = []
            for inner_loader in buffer:
                jax, _ = self.latent_statistics_calculator.encoding_model.calculate_jacobian(
                    loader=inner_loader,
                )
                if self.verbose > 0:
                    jax = tqdm(jax, desc=f"Calculating jacobian for loader in buffer")
                for jac_batch in jax:
                    for i, jac in enumerate(jac_batch):
                        if self.verbose > 0:
                            jax.set_description(f"calculating chunk [{i+1}/{len(jac_batch)}]")
                        jac = jac.to(self.likelihood_model.device)
                        jtj_eigvals = torch.linalg.eigvalsh(jac.T @ jac).cpu().numpy().flatten()
                        all_jtj_eigvals.append(jtj_eigvals)
            
                all_jtj_eigvals = np.stack(all_jtj_eigvals, axis=0)
                ref_jtj_eigvals_mean = np.mean(all_jtj_eigvals, axis=0)
                ref_jtj_eigvals_std = np.std(all_jtj_eigvals, axis=0)
                ref_jtj_eigvals_mean = torch.from_numpy(ref_jtj_eigvals_mean).to(self.likelihood_model.device)
                ref_jtj_eigvals_std = torch.from_numpy(ref_jtj_eigvals_std).to(self.likelihood_model.device)
                # store the extrinsic dimension
                D = len(ref_jtj_eigvals_mean)
                
                if self.adaptive_measurement:
                    self.evaluate_r = ref_jtj_eigvals_mean[10]
                    if self.log_scale:
                        self.evaluate_r = np.log(self.evaluate_r)
                else:
                    if self.automatic_jacobian_scaling:
                        def f(x, use_cache=True):
                            dimensionalities =  self.latent_statistics_calculator.calculate_statistics(
                                self.evaluate_r,
                                loader=inner_loader,
                                use_cache=use_cache,
                                log_scale=self.log_scale,
                                order=1,
                                ref_eigval_mean=ref_jtj_eigvals_mean,
                                ref_eigval_std=ref_jtj_eigvals_std,
                                scaling_factor=x,
                            )
                            msk1 = dimensionalities < D / 2
                            msk2 = dimensionalities > 20
                            return np.sum(msk1) + np.sum(msk2)
                        
                        L = 1.0
                        _ = f(1.0, use_cache=False)
                        R = max(1.0, L + self.jacobian_relative_scaling_factor)
                        
                        # perform ternery search to find the scaling factor that produces the most
                        # variance for in-distribution dimensionalities
                        for _ in tqdm(range(20), desc="calculating automatic scaling factor", total=20):
                            left_third = (2 * L + R) / 3
                            right_third = (L + 2 * R) / 3
                            
                            l_val = f(left_third)
                            r_val = f(right_third)
                            
                            if l_val < r_val:
                                L = left_third
                            else:
                                R = right_third
                        scaling_factor = (L + R) / 2
                    else:
                        scaling_factor = self.jacobian_relative_scaling_factor
            
            
            calculate_statistics_additional_args['scaling_factor'] = scaling_factor
            calculate_statistics_additional_args['ref_eigval_mean'] = ref_jtj_eigvals_mean
            calculate_statistics_additional_args['ref_eigval_std'] = ref_jtj_eigvals_std
        
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
        
        buff = 0
        for inner_loader in buffer:
            buff += 1
            inner_dimensionalities  = self.latent_statistics_calculator.calculate_statistics(
                self.evaluate_r,
                loader=inner_loader,
                use_cache=False,
                log_scale=self.log_scale,
                order=1,
                **calculate_statistics_additional_args,
            )
            all_dimensionalities = np.concatenate([all_dimensionalities, inner_dimensionalities]) if all_dimensionalities is not None else inner_dimensionalities
            
            
            for x in inner_loader:
                with torch.no_grad():
                    likelihoods = self.likelihood_model.log_prob(x).cpu().numpy().flatten()
                    all_likelihoods = np.concatenate([all_likelihoods, likelihoods]) if all_likelihoods is not None else likelihoods
        
        
        visualize_scatterplots(
            scores = np.stack([all_likelihoods, all_dimensionalities]).T,
            column_names=["log-likelihood", "dimensionality (d-D)"],
        )
    
class RadiiTrend(LatentBaseMethod):
    """
    Instead of evaluating the OOD data using a single score visualization,
    these methods evaluate the OOD data by visualizing the trend of the scores.
    
    Args:
        EllipsoidBaseMethod (_type_): _description_
    """
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        
        x: th.Optional[torch.Tensor] = None,
        x_batch: th.Optional[torch.Tensor] = None,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        logger: th.Optional[th.Any] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        
        
        # Latent statistics calculator
        latent_statistics_calculator_class: th.Optional[str] = None,
        latent_statistics_calculator_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        calculate_statistics_extra_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        # for logging args
        verbose: int = 0,
        
        # The range of the radii to show in the trend
        radii_range: th.Optional[th.Tuple[float, float]] = None,
        radii_n: int = 100,
        log_scale: bool = False,
        
        # visualization arguments
        visualization_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        # include reference or not
        include_reference: bool = False,
        
        compress_trend: bool = False,
        compression_bucket_size: th.Optional[int] = None,
        
        loader_buffer_size: int = 60,
        
        sampling: bool = False,
        sampling_args: th.Optional[th.Dict[str, th.Any]] = None,
    ):
        """
        Inherits from EllipsoidBaseMethod and adds the visualization of the trend.
        
        Args:
            radii_range (th.Optional[th.Tuple[float, float]], optional): The smallest and largest distance to consider.
            radii_n (int, optional): The number of Radii scales in the range to consider. Defaults to 100.
            visualization_args (th.Optional[th.Dict[str, th.Any]], optional): Additional arguments that will get passed on
                visualize_trend function. Defaults to None which is an empty dict.
            include_reference (bool, optional): If set to True, then the training data trend will also be visualized.
        """
        self.latent_statistics_calculator: LatentStatsCalculator
        
        super().__init__(
            likelihood_model=likelihood_model,
            x=x,
            x_batch=x_batch,
            x_loader=x_loader,
            logger=logger,
            in_distr_loader=in_distr_loader,
            latent_statistics_calculator_class=latent_statistics_calculator_class,
            latent_statistics_calculator_args=latent_statistics_calculator_args,
            verbose=verbose,
        )
        
        self.log_scale = log_scale
        if log_scale:
            self.radii = np.logspace(*np.log([r + 1e-12 for r in radii_range]), radii_n)
        else:
            self.radii = np.linspace(*radii_range, radii_n)
        self.calculate_statistics_extra_args = calculate_statistics_extra_args or {}
        
        self.visualization_args = visualization_args or {}
        
        self.include_reference = include_reference
        
        self.loader_buffer_size = loader_buffer_size
        
        
        self.compress_trend = compress_trend
        self.compression_bucket_size = compression_bucket_size
        
        self.sampling = sampling
        self.sampling_args = sampling_args or {}
            
    def calculate_radius_score(self, r, loader, use_cache: bool = False): 
        raise NotImplementedError("no calculate score defined!")
    
    def get_radius_score_name(self):
        return "score"
    
    def run(self):
        # This function first calculates the ellipsoids for each datapoint in the loader self.x_loader
        # This calculation is carried out with the ellipsoid calculator that is passed in the constructor.
        # Then, it increases the radius "r" to mintor how the CDF changes per datapoint
        # for visualization of multiple datapoints, the visualize_trend function is used.
        
        
        def get_trend(loader):
            
               
            # split the loader and calculate the trend
            trend = [] # This is the final trend
            trend_so_far = None # The is the cumulative uncompressed trend
            inner_loader_idx = 0
            for inner_loader in buffer_loader(loader, self.loader_buffer_size):
                
                inner_loader_idx += 1
                # display if verbose is set
                if self.verbose > 1:
                    radii_range = tqdm(self.radii)
                    
                    tot = (len(loader) + self.loader_buffer_size - 1) // self.loader_buffer_size
                        
                    radii_range.set_description(f"Calculating scores for loader in buffer [{inner_loader_idx}/{tot}]")
                else:
                    radii_range = self.radii
                
                inner_trend = []    
                first = True  
                for r in radii_range:
                    
                    if self.sampling:
                        imgs = self.latent_statistics_calculator.sample(
                            r, 
                            loader=loader, 
                            n_samples=9, 
                            use_cache=not first,
                            **self.sampling_args,
                        )
                        for i, im in enumerate(imgs):
                            # make a grid using torchvision
                            img_grid = torchvision.utils.make_grid(im, nrow=3)
                            with torch.no_grad():
                                im_ = im.to(self.likelihood_model.device)
                                # Replace all the NaN and inf values in im_ with 0
                                im_ = torch.where(torch.isnan(im_), torch.zeros_like(im_), im_)
                                im_ = torch.where(torch.isinf(im_), torch.zeros_like(im_), im_)
                                log_probs = self.likelihood_model.log_prob(im_)
                                log_probs = log_probs.cpu()
                            wandb.log(
                                {
                                    f"trend/vicinity_samples/{i+1}": [wandb.Image(
                                    img_grid, caption=f"samples with distance {r}\nexpected log_prob: {log_probs.mean().item()}")]
                                }
                            )
                        first = False
                    single_scores = self.latent_statistics_calculator.calculate_statistics(
                        np.log(r) if self.log_scale else r, 
                        loader=inner_loader, 
                        use_cache=not first,
                        log_scale=self.log_scale,
                        **self.calculate_statistics_extra_args,
                    )
                    inner_trend.append(
                        single_scores
                    )
                    first = False
                inner_trend = np.stack(inner_trend).T
                
                if trend_so_far is None:
                    trend_so_far = inner_trend
                else:
                    trend_so_far = np.concatenate([trend_so_far, inner_trend], axis=0)
                
                if self.compress_trend and trend_so_far.shape[0] >= self.compression_bucket_size:
                    # replace each column of trend with its mean
                    for l in range(0, trend_so_far.shape[0], self.compression_bucket_size):
                        r = l + self.compression_bucket_size
                        if r <= trend_so_far.shape[0]:
                            trend.append(np.mean(trend_so_far[l:r], axis=0))
                            
                    if r == trend_so_far.shape[0]:
                        trend_so_far = None
                    else:
                        trend_so_far = trend_so_far[r - self.compression_bucket_size:]
            
            if trend_so_far is not None:
                if self.compress_trend:
                    cmp_size = min(self.compression_bucket_size, trend_so_far.shape[0])
                else:
                    cmp_size = 1
                # replace each column of trend with its mean
                for l in range(0, trend_so_far.shape[0], cmp_size):
                    r = min(l + cmp_size, trend_so_far.shape[0])
                    trend.append(np.mean(trend_so_far[l:r], axis=0))
            
            return np.stack(trend, axis=0)
        
        if self.verbose > 0:
            print("Calculating trend ...")
            
        trend = get_trend(self.x_loader)
        
        # add reference if the option is set to True
        reference_trend = None
        if self.include_reference:
            
            if self.verbose > 0:
                print("Calculating reference trend ...")
            
            reference_trend = get_trend(self.in_distr_loader)
                  
        visualize_trends(
            scores=trend,
            t_values=np.log(self.radii) if self.log_scale else self.radii,
            reference_scores=reference_trend,
            x_label="log r" if self.log_scale else "r",
            y_label=self.latent_statistics_calculator.get_label(),
            title=f"Trend of the average {self.latent_statistics_calculator.get_name()}",
            **self.visualization_args,
        )

class GaussianConvolutionAnalysis(LatentBaseMethod):
    """
    This function calculates the convolution with the analytical approach
    and comes up with a single score based on the "rate" of change in the
    convolved density function.
    
    This rate is either empirically calculated or analytically done.
    
    By the end, this rate is plotted against the likelihood scores themselves in a scatterplot.
    """
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        
        x: th.Optional[torch.Tensor] = None,
        x_batch: th.Optional[torch.Tensor] = None,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        logger: th.Optional[th.Any] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        
        
        # Latent statistics calculator
        latent_statistics_calculator_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        # for logging args
        verbose: int = 0,
        
        # visualization arguments
        visualization_args: th.Optional[th.Dict[str, th.Any]] = None,
            
        loader_buffer_size: int = 60,
        
        evaluate_r: float = 1e-6,
        log_scale: bool = True,
        
        analysis_order: int = 2,
    ):
        """
        Inherits from EllipsoidBaseMethod and adds the visualization of the trend.
        
        Args:
            radii_range (th.Optional[th.Tuple[float, float]], optional): The smallest and largest distance to consider.
            radii_n (int, optional): The number of Radii scales in the range to consider. Defaults to 100.
            visualization_args (th.Optional[th.Dict[str, th.Any]], optional): Additional arguments that will get passed on
                visualize_trend function. Defaults to None which is an empty dict.
            include_reference (bool, optional): If set to True, then the training data trend will also be visualized.
        """
        self.latent_statistics_calculator: LatentStatsCalculator
        
        
        super().__init__(
            likelihood_model=likelihood_model,
            x=x,
            x_batch=x_batch,
            x_loader=x_loader,
            logger=logger,
            in_distr_loader=in_distr_loader,
            latent_statistics_calculator_class='ood.methods.linear_approximations.GaussianConvolutionRateStatsCalculator',
            latent_statistics_calculator_args=latent_statistics_calculator_args,
            verbose=verbose,
        )
        
        
        self.visualization_args = visualization_args or {}
        
        
        self.loader_buffer_size = loader_buffer_size
        
        self.log_scale = log_scale
        self.evaluate_r = log(evaluate_r) if self.log_scale else evaluate_r
        
        self.analysis_order = analysis_order
        if analysis_order < 2:
            raise ValueError("The analysis order must be at least 2!")
        
    def run(self):
        loader = self.x_loader
        
        all_convs = None
        
        if self.verbose > 0:
            buffered = tqdm(buffer_loader(loader, self.loader_buffer_size), desc="Calculating latent statistics for buffer")
        else:
            buffered = buffer_loader(loader, self.loader_buffer_size)
            
        for inner_loader in buffered:
            
            conv = []
            use_cache = False
        
            for ord in range(self.analysis_order):
                
                new_diffs = self.latent_statistics_calculator.calculate_statistics(
                    self.evaluate_r,
                    loader=inner_loader,
                    use_cache=use_cache,
                    order=ord,
                    log_scale=self.log_scale,
                )
                use_cache = True
                
                conv.append(new_diffs)
            conv = np.stack(conv, axis=1)
                
            all_convs = conv if all_convs is None else np.concatenate([all_convs, new_diffs])

        
        visualize_scatterplots(
            all_convs,
            column_names=[f"rate-{i}" for i in range(self.analysis_order)],
            title="rate-analysis",
        )
        
        

class LatentScore(LatentBaseMethod):
    """
    This set of methods summarizes the trends observed from R^2 into a single
    score that is used for OOD detection. This score can be obtained by different
    heuristics, which maybe choosed between by the user.
    """
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        
        x: th.Optional[torch.Tensor] = None,
        x_batch: th.Optional[torch.Tensor] = None,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        logger: th.Optional[th.Any] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        
        # Latent statistics arguments
        latent_statistics_calculator_class: th.Optional[str] = None,
        latent_statistics_calculator_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        # for logging args
        verbose: int = 0,
        
        # for visualization args
        visualization_args: th.Optional[th.Dict[str, th.Any]] = None,
        # metric
        metric_name: th.Optional[str] = None,
        metric_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        loader_buffer_size: int = 60,
    ):
        """
        This class adds the score calculation functionalities to the EllipsoidBaseMethod. By setting a metric name and a set of arguments
        you determine what kind of metric you want to calculate on top of R^2 to obtain a single score.
        
        Moreover, the single score would be visualized in W&B using histograms so you can easily compare the result of your run with other
        different OOD settings.
        
        Args:
            metric_name: Determine how the score is going to get computed. The following are currently implemented:
                1. 'cdf' for a predefined 'R'
                2. 'auto_cdf', where we first calculate an appropriate R by 
                    just looking at train data and then plugging it in for the CDF method in 1.
                3. 'cdf_inv' for a predefined 'q' (quantile) we calculate the R corresponding to that
                    using a binary search.
                4. 'first_moment': Calculate the average of R^2 which is the semantic distance in a 
                    fast and efficient manner (This is the fastest metric).
            metric_args: Additional arguments that the corresponding functions might need. Take a look at the implementations to see
                what are the appropriate arguments to pass in your configuration file.
        """
            
        super().__init__(
            likelihood_model=likelihood_model,
            x=x,
            x_batch=x_batch,
            x_loader=x_loader,
            logger=logger,
            in_distr_loader=in_distr_loader,
            latent_statistics_calculator_class=latent_statistics_calculator_class,
            latent_statistics_calculator_args=latent_statistics_calculator_args,
            verbose=verbose,
        )
        self.metric_name = metric_name
        self.metric_args = metric_args or {}

        self.visualization_args = visualization_args or {}
        
        self.loader_buffer_size = loader_buffer_size
        
    def _predef_r_score(self, r: float):
        """
        Given a specific 'r', it first calculates the entailed generalized-chi-squared
        distribution from self.x_loader, and then, it calculates the CDF of each of the
        ellipsoids.
        """
        scores = []
        for inner_loader in buffer_loader(self.x_loader, self.loader_buffer_size):
            scores.append(self.latent_statistics_calculator.calculate_statistics(r, loader=inner_loader, use_cache=False))
        return np.concatenate(scores)
    
    def _auto_r_score(
        self, 
        bn_search_limit: int = 100, 
        score_threshold: float = 0.001, 
        reference_loader_limit: th.Optional[int] = None
    ):
        """
        This function does a binary search to find the smallest 'r' in which the
        average cdf of the train data is larger than the cdf_threshold. This represents
        a meaningful semantic distance that can be used for OOD detection. Smaller values
        would have very tiny probabilities, while larger values might lose the local properties
        of R^2 that only hold for a small neighborhood.
        
        Args:
            bn_search_limit (int, optional): 
                For calculating the R, we do a binary search and this parameter determines
                the number of iterartions, which translates to the precision of the R. Defaults to 100.
            cdf_threshold (float, optional): 
                This is the infinitesimal threshold that is considered (meaningful).
                Defaults to 0.001.

        Returns:
            np.ndarray: The CDF of each of the ellipsoids ran on x_loader, with the r value
                obtained from the search above.
        """
        auto_r = 0
        idx = 0
            
        for inner_loader in buffer_loader(self.in_distr_loader, self.loader_buffer_size, limit=reference_loader_limit):
            r_l = 0.0
            r_r = 1e12
            
            if self.verbose == 1:
                rng = tqdm(range(bn_search_limit))
                rng.set_description("Binary Searching for the best R")
            else:
                rng = range(bn_search_limit)
            
            first = True
            for iter in rng:
                mid = (r_l + r_r) / 2.0
                if self.verbose > 1:
                    print(f"Binary Searching iteration no.{iter + 1}")
                elif self.verbose == 1:
                    rng.set_description(f"Binary Searching iteration no.{iter + 1}")
                
                single_scores = self.latent_statistics_calculator.calculate_statistics(mid, loader=inner_loader, use_cache=not first)
                if np.mean(single_scores) > score_threshold:
                    r_r = mid
                else:
                    r_l = mid
                    
                first = False
            
            auto_r = (idx * auto_r + r_r) / (idx + 1)
            idx += 1
            
        if self.verbose > 1:
            print(f"Found r = {auto_r}")
            
        return self._predef_r_score(auto_r)
    
    def _score_inv(self, q: float, bn_search_limit: int = 100):
        """
        For each datapoint in self.x_loader, we calculate the CDF inverse of the
        entailed semantic distance from that datapoint. To calculate the inverse,
        we employ a binary search to find the smallest 'r' in which the cdf of that 
        datapoint is larger than q.
        """
        
        scores = []
        for inner_loader in buffer_loader(self.x_loader, self.loader_buffer_size):
            T = sum([x.shape[0] for x in inner_loader])
           
            r_l = np.zeros(T)
            r_r = np.ones(T) * 1e12
            
            if self.verbose == 1:
                rng = tqdm(range(bn_search_limit))
            else:
                rng = range(bn_search_limit)
            
            first = True
            for iter in rng:
                if self.verbose > 1:
                    print(f"Binary Searching iteration no.{iter + 1}")
                if self.verbose == 1:
                    rng.set_description(f"Binary Searching iteration no.{iter + 1}")
                    
                mid = (r_l + r_r) / 2.0
                single_scores = self.latent_statistics_calculator.calculate_statistics(mid, loader=inner_loader, use_cache=not first)
                first = False
                r_l = np.where(single_scores > q, r_l, mid)
                r_r = np.where(single_scores > q, mid, r_r)
            scores.append(r_r)
        return np.concatenate(scores)
    
    def run(self):
        metric_name_to_func = {
            "auto-r-score": self._auto_r_score,
            "score-inv": self._score_inv,
            "predef-r-score": self._predef_r_score,
        }
        x_label = {
            "auto-r-score": f"auto-{self.latent_statistics_calculator.get_label()}",
            "score-inv": f"{self.latent_statistics_calculator.get_label()}-inv",
            "predef-r-score": f"{self.latent_statistics_calculator.get_label()}",
        }
        scores = metric_name_to_func[self.metric_name](**self.metric_args)
        
        # add the scores as a table to weights and biases
        tbl = wandb.Table(data=[[x] for x in scores], columns = [self.metric_name])
        wandb.log({
            f"native_histogram/{self.metric_name} on R^2": wandb.plot.histogram(tbl, self.metric_name, title=f"{self.metric_name} on R^2")
        })
        
        visualize_histogram(
            scores,
            title=f"{self.metric_name} on R^2",
            x_label=x_label[self.metric_name],
            y_label="Density",
            **self.visualization_args,
        )
