from .base_method import OODBaseMethod
import typing as th
import torch
import math
from ..visualization import visualize_histogram, visualize_trends
import numpy as np
from chi2comb import ChiSquared, chi2comb_cdf
from tqdm import tqdm
import wandb
import dypy as dy

class EncodingModel:
    """
    This is a class that is used for likelihood models that have an encode and
    decode functionality with them. Here, the encode and decode function are 
    abstract. For example, for a flow model, that would be running the transformation.
    And for a VAE model, that would be running the encoder and the decoder.

    Moreover, this class has a set of knobs designed for optimizing the computation
    of the Jacobian.
    """
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        # These are computation-related knobs
        # For example, with use_functorch and use_forward_mode
        # you can specify how to compute the Jacobian
        # With computation_batch_size you determine how large your jacobian batch
        # is going to be
        # With loader_limit you can specify how many batches you want to use
        use_vmap: bool = False,
        use_functorch: bool = True,
        use_forward_mode: bool = True,
        computation_batch_size: th.Optional[int] = None,
        loader_limit: th.Optional[int] = None,
        # verbosity
        verbose: int = 0,
    ) -> None:
        
        self.likelihood_model = likelihood_model
        
        # Set the model to evaluation mode
        self.likelihood_model.eval()
        
        self.use_functorch = use_functorch
        self.use_forward_mode = use_forward_mode
        self.use_vmap = use_vmap
        self.computation_batch_size = computation_batch_size
        
        if loader_limit is None:
            self.loader_limit = math.inf
        else:
            self.loader_limit = loader_limit
        
        self.verbose = verbose
    
    def decode(self, z):
        """Decodes the input from a standard Gaussian latent space 'z' onto the data space."""
        raise NotImplementedError("You must implement a decode method for your model.")
    
    def encode(self, x):
        """Encodes the input 'x' onto a standard Gaussian Latent Space."""
        raise NotImplementedError("You must implement an encode method for your model.")
    
    def calculate_jacobian(self, loader):
        """
        This function takes in a loader of the form
        [
            (x_batch_1, _, _),
            (x_batch_2, _, _),
            ...,
            (x_batch_n, _, _)
        ]
        where each x_batch_i is a batch of data points.
        
        This function calculates the encoding of each of the datapoints,
        then using the latent representation, calculates the Jacobian of
        the decoded function. This is done batch-wise to optimize the computation.
        """
        
        idx = 0
        
        if self.verbose > 1:
            loader_decorated = tqdm(loader)
        else:
            loader_decorated = loader
        
        jax = []
        z_values = []
           
        for x_batch, _, _ in loader_decorated:
            idx += 1
            if idx > self.loader_limit:
                # if loader limit is set, then
                # we won't need to calculate 
                # a lot of batches
                break
            
            jac = []
            # encode to obtain the latent representation in the Gaussian space
            z = self.encode(x_batch)
            
            # Since Jacobian computation is heavier than the normal batch_wise
            # computations, we have another level of batching here
            step = self.computation_batch_size
            if self.computation_batch_size is None:
                step = z.shape[0]
            
            progress = 0
            for l in range(0, z.shape[0], step):
                progress += 1
                if self.verbose > 1:
                    loader_decorated.set_description(f"Computing latent jacobians [{progress}/{z.shape[0] // step}]")
        
                r = min(l + step, z.shape[0])
                
                # Get a batch of latent representations
                z_s = z[l:r]

                # Calculate the jacobian of the decode function
                if self.use_functorch:
                    jac_fn = torch.func.jacfwd if self.use_forward_mode else torch.func.jacrev
                    if self.use_vmap:
                        # optimized implementation with vmap, however, it does not work as of yet
                        jac_until_now = torch.func.vmap(jac_fn(self.decode))(z_s)
                    else:
                        jac_until_now = jac_fn(self.decode)(z_s)
                else:
                    jac_until_now = torch.autograd.functional.jacobian(self.decode, z_s)

                # Reshaping the jacobian to be of the shape (batch_size, latent_dim, latent_dim)
                if self.use_vmap:
                    jac_until_now = jac_until_now.reshape(z_s.shape[0], -1, z_s.numel() // z_s.shape[0])
                    for j in range(jac_until_now.shape[0]):
                        jac.append(jac_until_now[j, :, :])
                else:
                    jac_until_now = jac_until_now.reshape(z_s.shape[0], -1, z_s.shape[0], z_s.numel() // z_s.shape[0])
                    for j in range(jac_until_now.shape[0]):
                        jac.append(jac_until_now[j, :, j, :])   
                    
                        
            # Stack all the obtained jacobians from the computation batch scale
            # to original batch scale
            jac = torch.stack(jac)
            
            jax.append(jac.cpu().detach())
            z_values.append(z.cpu().detach())
        
        return jax, z_values

class CDFCalculator:
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        # Encoding and decoding model
        encoding_model_class: th.Optional[str] = None, 
        encoding_model_args: th.Optional[th.Dict[str, th.Any]] = None,
        # verbosity
        verbose: int = 0,
    ):
        self.verbose = verbose
        self.encoding_model = dy.eval(encoding_model_class)(likelihood_model, **(encoding_model_args or {}))
    
    def calculate_single_cdf(self, r, loader, use_cache: bool = False) -> np.ndarray:
        # You should implement your own function here
        raise NotImplementedError("You must implement a calculate_single_cdf method for your model.")

    def calculate_mean_cdf(self, r, loader, use_cache: bool = False) -> float:
        """
        Passes the exact arguments to calculate_single_cdf and returns the mean of the cdfs
        which will also represent a complex semantic distance CDF.
        """
        if self.verbose > 1:
            print("Calculating the mean CDF...")
        return np.mean(self.calculate_single_cdf(r, loader, use_cache))

class CDFElliposoidCalculator(CDFCalculator):
    """
    This is a class that is used in methods that use latent ellipsoids.
    For a likelihood model that has an encoding and a decoding into a Gaussian
    space, this ellipsoid calculator yields all the calculations required to 
    calculate the corresponding ellipsoid for each data point.
    
    This class has an abstract encode and decode method that should be implemented for
    specific generative models. For example, for a flow model, that would be running
    the transformation both ways, and for a VAE, that would be running the encoder and
    the decoder.
    
    Moreover, we have computation knobs to optimize the computation of the Jacobian.
    For example, one might use vmaps to optimize the computation of the Jacobian of a 
    batch. Or one can use either functorch or autograd to compute the Jacobian.
    """
    def __init__(
        self,
        *args,
        lim: int = 1000,
        atol: float =1e-4,
        marginalization_threshold: float = 1e-4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.lim = lim
        self.atol = atol
        self.marginalization_threshold = marginalization_threshold
        
    def calculate_ellipsoids(self, loader):
        """
        This function takes in a loader of the form
        [
            (x_batch_1, _, _),
            (x_batch_2, _, _),
            ...,
            (x_batch_n, _, _)
        ]
        where each x_batch_i is a batch of data points.
        
        This function calculates the ellipsoids for each of the data points in the batch.
        And optimizes some of the computations by doing either vmap or batching if specified.
        After that, it returns the centers and radii of the ellipsoids in the following format.
        
        centers = [centers_batch_1, centers_batch_2, ..., centers_batch_n]
        radii = [radii_batch_1, radii_batch_2, ..., radii_batch_n]
        
        Where each of the centers_batch_i and radii_batch_i 
        have the shape (batch_size, latent_dim)
        """
        centers = []
        radii = []
        
        jax, z_values = self.encoding_model.calculate_jacobian(loader)
        
        if self.verbose > 1:
            iterable = tqdm(zip(jax, z_values))
            iterable.set_description("Calculating the ellipsoids")
        else:
            iterable = zip(jax, z_values)
        
        for jac, z in iterable: 
            ellipsoids = torch.matmul(jac.transpose(1, 2), jac)
            
            L, Q = torch.linalg.eigh(ellipsoids)
            # Each row of L would contain the scales of the non-central chi-squared distribution
            center = torch.matmul(Q.transpose(1, 2), z.unsqueeze(-1)).squeeze(-1)
            
            centers.append(center.cpu().detach())
            radii.append(L.cpu().detach())
            
        return centers, radii

    def calculate_single_cdf(self, r, loader, use_cache: bool = False) -> np.ndarray:
        """
        Given two loaders of radii and centers, this function calculates the
        CDF of each of the entailed generalized chi-squared distributions.
        If use_cache is set to true, then this means that centers and radii have been
        cached from before and some of the computations can be optimized.
        
        Args:
            r: Union(float, List[float]) 
                r value(s) where the cdf is evaluated.
            centers: The loader for the centers of the ellipsoids.
                [batch_centers_1, batch_centers_2, ..., batch_centers_n]
            radii: The loader for the radius of the ellipsoids.
                [batch_radii_1, batch_radii_2, ..., batch_radii_n]
            use_cache: (bool)
                Whether to use the cached centers and radii or not.
        Returns:
            np.ndarray: The CDF of each of the ellipsoids when the entire loader
                is flattened. In other words, if batch_i has size sz_i and T = sum(sz_i),
                then the returned array would have size T.
        """
        if not use_cache:
            self.centers, self.radii = self.calculate_ellipsoids(loader)
        if not hasattr(self, 'centers') or not hasattr(self, 'radii'):
            raise ValueError("You should first calculate the ellipsoids before calculating the CDFs with cache.")
        centers, radii = self.centers, self.radii
        def _check_r_primitive():
            if isinstance(r, float):
                return True
            if isinstance(r, int):
                return True
        if self.verbose > 1:
            print("Calculating the single CDFs...")
            if _check_r_primitive():
                print(f"\tr = {r}")
            
        if not use_cache or not hasattr(self, 'chi2s'):
            # if it is not cached, then we should calculate the chi2s
            self.chi2s = []
            for c_batch, rad_batch in zip(centers, radii):
                for c, rad in zip(c_batch, rad_batch):
                    chi2s_ = []
                    for c_i, r_i in zip(c, rad):
                        if r_i > self.marginalization_threshold:
                            chi2s_.append(ChiSquared(coef=r_i, ncent=c_i**2, dof=1))
                    if len(chi2s_) == 0:
                        raise Exception("Marginalization threshold is too large; no coordinates remain!")
                    self.chi2s.append(chi2s_)
        
        
        if self.verbose > 1:
            rng_chi2s = tqdm(self.chi2s)
            rng_chi2s.set_description("Calculating the CDFs for each of the chi2s")
        else:
            rng_chi2s = self.chi2s
                
        # self.chi2s contains a list of flattened chi2s
        cdfs = []
        if _check_r_primitive():
            for chi2s in rng_chi2s:
                cdfs.append(chi2comb_cdf(r**2, chi2s, 0.0, lim=self.lim, atol=self.atol)[0])
        else:
            for r_single, chi2s in zip(r, rng_chi2s):
                cdfs.append(chi2comb_cdf(r_single**2, chi2s, 0.0, lim=self.lim, atol=self.atol)[0])
                
        return np.array(cdfs)

    def calculate_mean(self, loader, use_cache=False):
        if not use_cache:
            self.centers, self.radii = self.calculate_ellipsoids(loader)
        scores = []
        for c_batch, rad_batch in zip(self.centers, self.radii):
            for c, rad in zip(c_batch, rad_batch):
                scores.append(np.sum(rad.numpy() * (1 + c.numpy()**2)))
        return np.array(scores)
    

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
    
    def run(self):
        # This function first calculates the ellipsoids for each datapoint in the loader self.x_loader
        # This calculation is carried out with the ellipsoid calculator that is passed in the constructor.
        # Then, it increases the radius "r" to mintor how the CDF changes per datapoint
        # for visualization of multiple datapoints, the visualize_trend function is used.
        trend = []
        
        # display if verbose is set
        if self.verbose > 1:
            radii_range = tqdm(self.radii)
            radii_range.set_description("Calculating the trend of the average cdf")
        else:
            radii_range = self.radii
        
        first = True  
        for r in radii_range:
            trend.append(self.cdf_calculator.calculate_single_cdf(r, loader=self.x_loader, use_cache=not first))
            first = False
            
        trend = np.stack(trend).T
        
        # add reference if the option is set to True
        reference_scores = None
        if self.include_reference:
            reference_scores = []
            
            if self.verbose > 1:
                radii_range = tqdm(self.radii)
                radii_range.set_description("Calculating the trend of the average cdf for reference")
            else:
                radii_range = self.radii
            
            first = True
            for r in radii_range:
                reference_scores.append(self.cdf_calculator.calculate_single_cdf(r, loader=self.in_distr_loader, use_cache=not first))
                first = False
            reference_scores = np.stack(reference_scores).T
                  
        visualize_trends(
            scores=trend,
            t_values=self.radii,
            reference_scores=reference_scores,
            x_label="r",
            y_label="CDF(R^2(r^2))",
            title="Trend of the average cdf",
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
        
    def _predef_r_cdf(self, r: float):
        """
        Given a specific 'r', it first calculates the entailed generalized-chi-squared
        distribution from self.x_loader, and then, it calculates the CDF of each of the
        ellipsoids.
        """
        return self.cdf_calculator.calculate_single_cdf(r, loader=self.x_loader, use_cache=False)
    
    def _find_r_then_cdf(self, bn_search_limit: int = 100, cdf_threshold: float = 0.001):
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
            
            if self.cdf_calculator.calculate_mean_cdf(mid, loader=self.in_distr_loader, use_cache=not first) > cdf_threshold:
                r_r = mid
            else:
                r_l = mid
                
            first = False
            
        auto_r = r_r
        
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
        
        # just a single dummy run    
        single_cdfs = self.cdf_calculator.calculate_single_cdf(0.0, loader=self.x_loader, use_cache=False)
        
        r_l = np.zeros(len(single_cdfs))
        r_r = np.ones(len(single_cdfs)) * 1e12
        
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
            single_cdfs = self.cdf_calculator.calculate_single_cdf(mid, loader=self.x_loader, use_cache=not first)
            first = False
            r_l = np.where(single_cdfs > q, r_l, mid)
            r_r = np.where(single_cdfs > q, mid, r_r)
        return r_r
    
    def _first_moment(self):
        """
        Calculate the first moment of the generalized chi-squared entailed by each data
        point in a flattened manner.
        
        Returns:
            np.ndarray: The average R^2.
        """
        return self.cdf_calculator.calculate_mean(loader=self.x_loader)
        
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

# TODO: Implement the VAE models here                
class EncodingVAE(EncodingModel):
    """
    The EncodingModel implemented for the VAEs in the model_zoo repository.
    """
    pass

class EncodingFlow(EncodingModel):
    """
    The EncodingModel implemented for the NormFlows in the model_zoo repository.
    """
    def encode(self, x):
        # turn off the gradient for faster computation
        # because we don't need to change the model parameters
        # self.likelihood_model.eval()
        with torch.no_grad():
            try:
                z = self.likelihood_model._nflow.transform_to_noise(x)
            except ValueError as e:
                z = self.likelihood_model._nflow.transform_to_noise(x.unsqueeze(0))
                z = z.squeeze(0)
            return z

    def decode(self, z):
        # turn off the gradient for faster computation
        # because we don't need to change the model parameters
        # self.likelihood_model.eval()
        with torch.no_grad():
            try:
                x, logdets = self.likelihood_model._nflow._transform.inverse(z)
            except ValueError as e:
                x, logdets = self.likelihood_model._nflow._transform.inverse(z.unsqueeze(0))
                x = x.squeeze(0)
            return x