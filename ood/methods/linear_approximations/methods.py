from ..base_method import OODBaseMethod
import typing as th
import torch
import math
from ...visualization import visualize_histogram, visualize_trends
import numpy as np
from chi2comb import ChiSquared, chi2comb_cdf
from tqdm import tqdm
import wandb
import dypy as dy
from scipy.stats import norm
from scipy.special import logsumexp
from .utils import buffer_loader


class CDFBaseMethod(OODBaseMethod):
    """
    This class is the base class that employs an ellipsoid calculator.
    
    All the methods that rely on these computations should inherit from this class.
    For example, the EllipsoidTrend method inherits this one and uses the average
    R^2 trend to detect or visualize OOD data. On the other hand, EllipsoidScore summarizes
    information from R^2 into a specific score and then uses that score plus thresholding
    to calculate OOD data.
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
        
        # CDF calculator
        cdf_calculator_class: th.Optional[str] = None,
        cdf_calculator_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        # for logging args
        verbose: int = 0,
    ):
        """
        Args:
            likelihood_model (torch.nn.Module): The likelihood model with initialized parameters that work best for generation.
            metric_name (str): The class of the metric in use.
            metric_args (th.Optional[th.Dict[str, th.Any]], optional): The instantiation arguments for the metric. Defaults to None which means empty dict.
            x (th.Optional[torch.Tensor], optional): If the OOD detection task is only aimed for single point, this is the point.
            x_batch (th.Optional[torch.Tensor], optional): _description_. Defaults to None.
            x_loader (th.Optional[torch.utils.data.DataLoader], optional): _description_. Defaults to None.
            logger (th.Optional[th.Any], optional): _description_. Defaults to None.
            in_distr_loader (th.Optional[torch.utils.data.DataLoader], optional): _description_. Defaults to None.
            verbose (int, optional): _description_. Defaults to 0.
            cdf_calculator_class: (str) The class of the CDF calculator
            cdf_calculator_args (dict) The arguments used for visualizing the cdf calculator
        """
        super().__init__(
            x_loader=x_loader, 
            x=x, 
            x_batch=x_batch, 
            likelihood_model=likelihood_model, 
            logger=logger, 
            in_distr_loader=in_distr_loader, 
        )
        
        cdf_calculator_args = cdf_calculator_args or {}
        if 'verbose' not in cdf_calculator_args:
            cdf_calculator_args['verbose'] = verbose
            
        self.cdf_calculator: CDFCalculator = dy.eval(cdf_calculator_class)(likelihood_model, **(cdf_calculator_args or {}))
        
        if self.x is not None:    
            self.x_batch = self.x.unsqueeze(0)
        
        if self.x_batch is not None:
            # create a loader with that single batch
            self.x_loader = [(self.x_batch, None, None)]
        
        self.verbose = verbose
        
class CDFTrend(CDFBaseMethod):
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
        
        
        # CDF calculator
        cdf_calculator_class: th.Optional[str] = None,
        cdf_calculator_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        # for logging args
        verbose: int = 0,
        
        # The range of the radii to show in the trend
        radii_range: th.Optional[th.Tuple[float, float]] = None,
        radii_n: int = 100,
        
        # visualization arguments
        visualization_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        # include reference or not
        include_reference: bool = False,
        
        compress_trend: bool = False,
        compression_bucket_size: th.Optional[int] = None,
        
        loader_buffer_size: int = 60,
        loader_limit: th.Optional[int] = None,
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
        self.cdf_calculator: CDFCalculator
        
        super().__init__(
            likelihood_model=likelihood_model,
            x=x,
            x_batch=x_batch,
            x_loader=x_loader,
            logger=logger,
            in_distr_loader=in_distr_loader,
            cdf_calculator_class=cdf_calculator_class,
            cdf_calculator_args=cdf_calculator_args,
            verbose=verbose,
        )
        
        self.radii = np.linspace(*radii_range, radii_n)
        
        self.visualization_args = visualization_args or {}
        
        self.include_reference = include_reference
        
        self.loader_buffer_size = loader_buffer_size
        self.loader_limit = loader_limit
        
        
        self.compress_trend = compress_trend
        self.compression_bucket_size = compression_bucket_size
        
        if 'visualize_log' in self.visualization_args:
            self.visualize_log = self.visualization_args.pop('visualize_log')
        else:
            self.visualize_log = False
            
    
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
            for inner_loader in buffer_loader(loader, self.loader_buffer_size, limit=self.loader_limit):
                
                inner_loader_idx += 1
                # display if verbose is set
                if self.verbose > 1:
                    radii_range = tqdm(self.radii)
                    
                    tot = (len(loader) + self.loader_buffer_size - 1) // self.loader_buffer_size
                    if self.loader_limit is not None:
                        tot = min(tot, self.loader_limit)
                        
                    radii_range.set_description(f"Calculating cdf for buffer [{inner_loader_idx}/{tot}]")
                else:
                    radii_range = self.radii
                
                inner_trend = []    
                first = True  
                for r in radii_range:
                    if self.visualize_log:
                        inner_trend.append(self.cdf_calculator.calculate_single_log_cdf(r, loader=inner_loader, use_cache=not first))
                    else:
                        inner_trend.append(self.cdf_calculator.calculate_single_cdf(r, loader=inner_loader, use_cache=not first))
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
            t_values=self.radii,
            reference_scores=reference_trend,
            x_label="r",
            y_label=f"{'log P' if self.visualize_log else 'P'}(R < r)",
            title=f"Trend of the average {'log_cdf' if self.visualize_log else 'cdf'}",
            **self.visualization_args,
        )
        
class CDFScore(CDFBaseMethod):
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
        
        # CDF calculator
        cdf_calculator_class: th.Optional[str] = None,
        cdf_calculator_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        # for logging args
        verbose: int = 0,
        
        # for visualization args
        visualization_args: th.Optional[th.Dict[str, th.Any]] = None,
        # metric
        metric_name: th.Optional[str] = None,
        metric_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        loader_limit: th.Optional[int] = None,
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
            cdf_calculator_class=cdf_calculator_class,
            cdf_calculator_args=cdf_calculator_args,
            verbose=verbose,
        )
        self.metric_name = metric_name
        self.metric_args = metric_args or {}

        self.visualization_args = visualization_args or {}
        
        self.loader_limit = loader_limit
        self.loader_buffer_size = loader_buffer_size
        
    def _predef_r_cdf(self, r: float):
        """
        Given a specific 'r', it first calculates the entailed generalized-chi-squared
        distribution from self.x_loader, and then, it calculates the CDF of each of the
        ellipsoids.
        """
        scores = []
        for inner_loader in buffer_loader(self.x_loader, self.loader_buffer_size, limit=self.loader_limit):
            scores.append(self.cdf_calculator.calculate_single_cdf(r, loader=inner_loader, use_cache=False))
        return np.concatenate(scores)
    
    def _find_r_then_cdf(
        self, 
        bn_search_limit: int = 100, 
        cdf_threshold: float = 0.001, 
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
        
        # use the loader_limit from the arguments if given
        # otherwise, use the default loader_limit
        if reference_loader_limit is None:
            reference_loader_limit = self.loader_limit
            
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
                
                if self.cdf_calculator.calculate_mean_cdf(mid, loader=inner_loader, use_cache=not first) > cdf_threshold:
                    r_r = mid
                else:
                    r_l = mid
                    
                first = False
            
            auto_r = (idx * auto_r + r_r) / (idx + 1)
            idx += 1
            
        if self.verbose > 1:
            print(f"Found r = {auto_r}")
            
        return self._predef_r_cdf(auto_r)
    
    def _cdf_inv(self, q: float, bn_search_limit: int = 100):
        """
        For each datapoint in self.x_loader, we calculate the CDF inverse of the
        entailed semantic distance from that datapoint. To calculate the inverse,
        we employ a binary search to find the smallest 'r' in which the cdf of that 
        datapoint is larger than q.
        """
        
        scores = []
        for inner_loader in buffer_loader(self.x_loader, self.loader_buffer_size, limit=self.loader_limit):
            T = sum([x.shape[0] for x, _, _ in inner_loader])
           
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
                single_cdfs = self.cdf_calculator.calculate_single_cdf(mid, loader=inner_loader, use_cache=not first)
                first = False
                r_l = np.where(single_cdfs > q, r_l, mid)
                r_r = np.where(single_cdfs > q, mid, r_r)
            scores.append(r_r)
        return np.concatenate(scores)
    
    def _first_moment(self):
        """
        Calculate the first moment of the generalized chi-squared entailed by each data
        point in a flattened manner.
        
        Returns:
            np.ndarray: The average R^2.
        """
        scores = []
        for inner_loader in buffer_loader(self.x_loader, self.loader_buffer_size, limit=self.loader_limit):
            scores.append(self.cdf_calculator.calculate_mean(loader=inner_loader))
        return np.concatenate(scores)
    
    def run(self):
        metric_name_to_func = {
            "find-r-then-cdf": self._find_r_then_cdf,
            "cdf-inv": self._cdf_inv,
            "predef-r-cdf": self._predef_r_cdf,
            "first-moment": self._first_moment,
        }
        x_label = {
            "find-r-then-cdf": "CDF",
            "cdf-inv": "CDF^{-1}",
            "predef-r-cdf": "CDF",
            "first-moment": "First Moment",
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
