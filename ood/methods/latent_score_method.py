"""
These methods only work on flow models.
They get the latent representation of the data and then compute a score based on that.

Some of the methods just consider the Lp norm of the latent representation of the data.
Some other consider the probability in an Lp ball of radius r around the latent representation of the data.
Others might incorporate semantic information in the local latent space; for example, they
might consider an ellipsoid around the latent representation of the data and then calculate
the probability measure of the ellipsoid. This ellipsoid might be semantically aware with
more variations on the dimensions that are more important for the data.
"""
import typing as th
import torch
from .base_method import OODBaseMethod
import dypy as dy 
import numpy as np
import wandb
from tqdm import tqdm
from scipy.stats import norm, ncx2
from chi2comb import chi2comb_cdf, ChiSquared
import time
import math
import dypy as dy
from ..visualization import visualize_histogram
from nflows import transforms, distributions, flows, utils


class LatentScorePrimitive(OODBaseMethod):
    """
    Calculates a score only based on the latent representation of the data.
    
    For example, if we are considering an Lp norm based method and p is given and the
    score is the norm, then it just calculates the norm of the latent representation 
    of the data. According to Nalisnick et al. (2019), the norm for p=2 of the latent 
    representation should fix the pathologies 
    
    TODO: for some reason we can't reproduce for CIFAR10-vs-SVHN
    """
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        
        
        
        x: th.Optional[torch.Tensor] = None,
        x_batch: th.Optional[torch.Tensor] = None,
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        logger: th.Optional[th.Any] = None,
        #
        score_type: th.Literal['norm', 'prob'] = 'norm',
        score_args: th.Optional[th.Dict[str, th.Any]] = None,
        # 
        progress_bar: bool = True,
        bincount: int = 5,
        
        **kwargs,
    ) -> None:
        super().__init__(x_loader=x_loader, x=x, x_batch=x_batch, likelihood_model=likelihood_model, logger=logger, **kwargs)
        
        if x is not None:
            self.x_batch = x.unsqueeze(0)
        
        self.progress_bar = progress_bar
        
        self.bincount = bincount
        
        self.score_type = score_type
        self.score_args = score_args if score_args is not None else {}
    
    def calc_score(self, z):
        if self.score_type == 'norm':
            p = self.score_args['p']
            if p == 'inf':
                return np.max(np.abs(z), axis=-1)
            else:
                return np.linalg.norm(z, ord=p, axis=-1)
        elif self.score_type == 'prob':
            radius = self.score_args['radius']
            p = self.score_args['p']
            eps_correction = self.score_args['eps_correction'] if 'eps_correction' in self.score_args else 1e-6
            if p == 'inf':
                log_score = 0
                for z_i in z:
                    r = z_i + radius
                    l = z_i - radius
                    # calculate the standard gaussian CDF of l and r
                    cdf_l = norm.cdf(l)
                    cdf_r = norm.cdf(r)
                    p_i = cdf_r - cdf_l
                    p_i = np.clip(p_i, a_min=eps_correction, a_max=1.0)
                    log_p_i = np.log(p_i)
                    log_score += log_p_i
                return log_score
            else:
                raise NotImplementedError('p != inf not implemented yet!')
        else:
            raise ValueError(f'Unknown score type {self.score_type}')
            
    def run(self):
        """
        Creates a histogram of scores, with the scores being the lp-norm of the latent representation of the data.
        """
        if not hasattr(self.likelihood_model, '_nflow'):
            raise ValueError('The likelihood model must have a _nflow attribute that returns the number of flows.')
        
            
        with torch.no_grad():
            all_scores = None
            if self.x_loader is not None:
                if self.progress_bar:
                    iterable = tqdm(self.x_loader)
                else:
                    iterable = self.x_loader
                for x_batch, _ in iterable:
                    z = self.likelihood_model._nflow.transform_to_noise(x_batch).cpu().detach().numpy()   
                    new_scores = self.calc_scores(z) 
                    all_scores = np.concatenate([all_scores, new_scores]) if all_scores is not None else new_scores
            else:
                z = self.likelihood_model._nflow.transform_to_noise(self.x_batch).cpu().detach().numpy()   
                all_scores = self.calc_score(z)

        visualize_histogram(
            all_scores,
            bincount=self.bincount,
            x_label=f'Score {self.score_type}',
            title=f'Score {self.score_type} histogram',
            y_label='Frequency',
        )
        # # create a density histogram out of all_scores
        # # and store it as a line plot in (x_axis, density)
        # hist, bin_edges = np.histogram(all_scores, bins=self.bincount, density=True)
        # density = hist / np.sum(hist)
        # centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # # get the average distance between two consecutive centers
        # avg_dist = np.mean(np.diff(centers))
        # # add two points to the left and right of the histogram
        # # to make sure that the plot is not cut off
        # centers = np.concatenate([[centers[0] - avg_dist], centers, [centers[-1] + avg_dist]])
        # density = np.concatenate([[0], density, [0]])
        
        # data = [[x, y] for x, y in zip(centers, density)]
        # table = wandb.Table(data=data, columns = ['score', 'density'])
        # wandb.log({'score_density': wandb.plot.line(table, 'score', 'density', title='Score density')})
    
    
class LpProbScore:
    """
    Calculates the probability of the latent representation of the data being in an Lp ball of radius r.
    """
    def __init__(
        self, 
        p: th.Union[float, int, th.Literal['inf']], 
        radius: float,
        eps_correction: float = 1e-6,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        likelihood_model: th.Optional[torch.nn.Module] = None,
    ) -> None:
        self.p = p
        self.radius = radius
        self.eps_correction = eps_correction
    
    def __call__(
        self, 
        z: torch.Tensor, 
        x: torch.Tensor, 
        likelihood_model
    ) -> float:
        z = z.cpu().detach().numpy() 
        radius = self.radius
        p = self.p
        eps_correction = self.eps_correction
        if p == 'inf':
            log_score = 0
            for z_i in z:
                r = z_i + radius
                l = z_i - radius
                # calculate the standard gaussian CDF of l and r
                cdf_l = norm.cdf(l)
                cdf_r = norm.cdf(r)
                p_i = cdf_r - cdf_l
                p_i = np.clip(p_i, a_min=eps_correction, a_max=1.0)
                log_p_i = np.log(p_i)
                log_score += log_p_i
            return log_score
        else:
            raise NotImplementedError('p != inf not implemented yet!')

class SemanticEllipsoidCDFDistance:
    """
    This score calculates the latent representation ellipsoid
    and considers a semantic distance R and computes the cdf of 
    average distance from in-distribution points.
    
    Then, it considers the point of interest and computes the distance
    R from that point and computes the distance between these two cdf
    functions.
    """
    def __init__(
        self, 
        n_radii: int = 100,
        limit_radii: float = 10,
        calculate_reference: bool = True,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        likelihood_model: th.Optional[torch.nn.Module] = None,
        computation_batch_size: int = 1,
        use_vmap: bool = False,
    ) -> None:
        self.likelihood_model = likelihood_model
        
        self.ambient_radii = np.linspace(0.0, limit_radii, n_radii)
        
        # determines the batch size of performing the computation in parallel
        self.computation_batch_size = computation_batch_size
        self.use_vmap = use_vmap
        
        if calculate_reference:
            # get a random batch from in_distr_loader
            in_batch, _, _ = next(iter(in_distr_loader))
            
            # cdf(r) for a single radius 'r' is charactarized by the
            # linear combination of a set of non-central chi-squared
            # distributions.
            
            # for a specific ellipse with radii <r_1, ..., r_n> and center <c_1, ..., c_n>
            # the RV is:
            # X = r_1 * \Chi_{c_1}^2 + ... + r_n * \Chi_{c_n}^2

            ellipsoid_cdf = np.zeros((in_batch.shape[0], n_radii))
            
            idx = 0
            radii_, centers_ = self._calculate_ellipsoids(in_batch)
            
            for radii, centers in zip(radii_, centers_):
                chi2s = [ChiSquared(coef=r_i, ncent=c_i**2, dof=1) for c_i, r_i in zip(centers, radii)]
                for r_idx, r in enumerate(self.ambient_radii):
                    ellipsoid_cdf[idx, r_idx] = chi2comb_cdf(r ** 2, chi2s, 0.0)[0]
                idx += 1
            
            self.reference_cdf = np.mean(ellipsoid_cdf, axis=0)
        else:
            self.reference_cdf = None
            
    def _calculate_ellipsoids(self, x, z=None):
        """
        get a batch of data and calculate the semantic ellipsoids 
        for those datapoints. 
        
        returns: (ellipsoid_radii, ellipsoid_centers)
            ellipsoid_radii: a tensor of shape (batch_size, n_radii)
            ellipsoid_centers: a tensor of shape (batch_size, c_i)
        """
        
        if z is None:
            # Encode to get the latent representation
            with torch.no_grad():
                z = self.likelihood_model._nflow.transform_to_noise(x)
        
        
        def get_ambient_from_latent(z):
            with torch.no_grad():
                try:
                    x, logdets = self.likelihood_model._nflow._transform.inverse(z)
                except ValueError as e:
                    x, logdets = self.likelihood_model._nflow._transform.inverse(z.unsqueeze(0))
                    x = x.squeeze(0)
            return x
        
        jac = []
        for i in tqdm(range(0, z.shape[0], self.computation_batch_size)):
            z_s = z[i:min(i+self.computation_batch_size, z.shape[0])]
            if self.use_vmap:
                # optimized implementation with vmap, however, it does not work as of yet
                jac_until_now = torch.func.vmap(torch.func.jacfwd(get_ambient_from_latent))(z_s)
                jac_until_now = jac_until_now.reshape(z_s.shape[0], -1, z_s.numel() // z_s.shape[0])
                for j in range(jac_until_now.shape[0]):
                    jac.append(jac_until_now[j, :, :])
            else:
                jac_until_now = torch.func.jacfwd(get_ambient_from_latent)(z_s)
                jac_until_now = jac_until_now.reshape(z_s.shape[0], -1, z_s.shape[0], z_s.numel() // z_s.shape[0])
                for j in range(jac_until_now.shape[0]):
                    jac.append(jac_until_now[j, :, j, :])
        jac = torch.stack(jac)
        # 
        # jac = torch.func.jacfwd(lambda z: self.likelihood_model._nflow._transform.inverse(z)[0])(z)
        
        ellipsoids = torch.matmul(jac.transpose(1, 2), jac)
        
        L, Q = torch.linalg.eigh(ellipsoids)
        
        rotated_z = torch.matmul(Q.transpose(1, 2), z.unsqueeze(-1)).squeeze(-1)
        
        # TODO: remove this: normalize each row of L
        L = L / L.sum(dim=-1, keepdim=True)
        
        return L, rotated_z
    
    def __call__(
        self, 
        z: torch.Tensor, 
        x: torch.Tensor, 
        likelihood_model
    ) -> float: 
        scores = []
        radii_, centers_ = self._calculate_ellipsoids(x, z)
        for radii, centers in zip(radii_, centers_):
            chi2s = [ChiSquared(coef=r_i, ncent=c_i**2, dof=1) for c_i, r_i in zip(centers, radii)]
            score = 0.0
            for i, r in enumerate(self.ambient_radii):
                cdf_r = chi2comb_cdf(r ** 2, chi2s, 0.0)[0]
                score += abs(self.reference_cdf[i] - cdf_r)
            scores.append(score)
        return np.array(scores) 
        

class SemanticAwareEllipsoidScore:
    """
    This score calculates the latent representation of the data.
    Then considers an allipsoid with it's radii being the magnitude of the 
    jacobian of the latent representation of the data. Then it calculates the 
    integral of the gaussian pdf over the ellipsoid analytically.
    """
    def __init__(
        self, 
        statistics: th.Literal['cdf-quantile', 'cdf-value', 'cdf', 'mean', 'mean-and-variance'] = 'cdf',
        quantile: float = 0.01,
        r_value: th.Optional[float] = None,
        radius: th.Optional[float] = None,
        n_radii: int = 10,
        p: th.Union[float, int, th.Literal['inf']] = 2,
        eps_correction: float = 1e-6,
        use_functorch: bool = True,
        use_forward_mode: bool = True,
        filtering_out: th.Optional[th.Union[str, th.Dict[str, str]]] = None,
        compute_radii: th.Optional[th.Union[str, th.Dict[str, str]]] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        likelihood_model: th.Optional[torch.nn.Module] = None,
    ) -> None:
        self.quantile = quantile
        self.p = p
        self.radius = radius
        self.eps_correction = eps_correction
        self.use_functorch = use_functorch
        self.use_forward_mode = use_forward_mode
        self.n_radii = n_radii
        self.statistics = statistics
        
        # a function that takes in eigenvalues and returns a boolean tensor
        # of the same size as eigenvalues. True means that the eigenvalue
        # should be filtered out.
        if filtering_out is None:
            # the function does not filter out anything and gets in a numpy array
            self.filtering_out = lambda x: np.zeros_like(x, dtype=bool)
        elif isinstance(filtering_out, str):
            self.filtering_out = dy.eval_function(filtering_out)
        elif isinstance(filtering_out, dict):
            self.filtering_out = dy.eval_function(**filtering_out)
        else:
            raise ValueError('filtering_out should be either None, a string or a dictionary.')
        
        # a function that takes in eigenvalues and computes the radii
        # of the corresponding ellipsoid.
        if compute_radii is None:
            # the function returns the magnitudes of the eigenvalues
            self.compute_radii = lambda x: np.abs(x)
        elif isinstance(compute_radii, str):
            self.compute_radii = dy.eval_function(compute_radii)
        elif isinstance(compute_radii, dict):
            self.compute_radii = dy.eval_function(**compute_radii)
        else:
            raise ValueError('compute_radii should be either None, a string or a dictionary.')
        
        self.r_value = r_value
    
    def __call__(
        self, 
        z: torch.Tensor, 
        x: torch.Tensor, 
        likelihood_model, 
        radius: th.Optional[th.Union[float, np.array]] = None
    ) -> float:
        
        
        for param in likelihood_model.parameters():
            param.requires_grad = False
        
        all_scores = []
        for z_i, x_i in tqdm(list(zip(z, x))):   
            def get_sample_from_latent(z):
                x_samples, logdets = likelihood_model._nflow._transform.inverse(z.unsqueeze(0))
                return x_samples.squeeze(0)
            
            if not self.use_functorch:
                jac = torch.autograd.functional.jacobian(get_sample_from_latent, z_i)
            else:
                if self.use_forward_mode:
                    jac = torch.func.jacfwd(get_sample_from_latent)(z_i)
                else:
                    jac = torch.func.jacrev(get_sample_from_latent)(z_i)
            
            
            jac = jac.reshape((x_i.numel(), z_i.numel())).cpu().detach().numpy()
            
            
            # calculate the eigenspectrum of jac
            eigvals, eigvecs = np.linalg.eigh(jac.T @ jac)
            center_rotated = eigvecs.T @ z_i.cpu().numpy()
            
            filtering_mask = self.filtering_out(eigvals)
            center_rotated = center_rotated[~filtering_mask]
            ellipsoid_radii = self.compute_radii(eigvals[~filtering_mask])
            
            # # make it relative (we're already playing around with the radius,
            # # no reason to have high ellipsoid_radii here. we just need to preserve
            # # relative scales between the radii)
            # ellipsoid_radii = ellipsoid_radii / np.sum(ellipsoid_radii)
            
            if self.p == 'inf':
                raise NotImplementedError('p = inf not implemented yet!')
            elif self.p == 2:
                # Dummy test of chi2comb_cdf
                # gcoef = 2
                # ncents = np.abs(center_rotated[:3])
                # q = 1
                # dofs = [1, 1, 1]
                # coefs = ellipsoid_radii[:3]
                # chi2s = [ChiSquared(coefs[i], ncents[i], dofs[i]) for i in range(3)]
                # result, errno, info = chi2comb_cdf(q, chi2s, gcoef)
                # print("Dummy result", result)

                
                
                if radius is not None:
                    chi2s = [ChiSquared(coef=r_i, ncent=c_i**2, dof=1) for c_i, r_i in zip(center_rotated, ellipsoid_radii)]
            
                    # check if center_rotated or ellipsoid_radii contains nan or inf
                    if np.any(np.isnan(center_rotated)) or np.any(np.isnan(ellipsoid_radii)):
                        raise ValueError("NaN in center_rotated or ellipsoid_radii")
                    
                    score = []
                    for r in radius:
                        cdf = chi2comb_cdf(r ** 2, chi2s, 0.0)[0]
                        # clamp the cdf to avoid numerical issues
                        score.append(np.clip(cdf, 0.0, 1.0))
                else:
                    if self.statistics == 'cdf':
                        chi2s = [ChiSquared(coef=r_i, ncent=c_i**2, dof=1) for c_i, r_i in zip(center_rotated, ellipsoid_radii)]
                        score = chi2comb_cdf(self.radius ** 2, chi2s, 0.0)[0]
                    elif self.statistics == 'cdf-value':
                        chi2s = [ChiSquared(coef=r_i, ncent=c_i**2, dof=1) for c_i, r_i in zip(center_rotated, ellipsoid_radii)]
                        score = chi2comb_cdf(self.r_value ** 2, chi2s, 0.0)[0]    
                    elif self.statistics == 'cdf-quantile':
                        chi2s = [ChiSquared(coef=r_i, ncent=c_i**2, dof=1) for c_i, r_i in zip(center_rotated, ellipsoid_radii)]
                        # do a binary search to find the first r which chi2comb_cdf(r**2, chi2, 0.0)[0]
                        # is at least 0.01
                        L = 0
                        R = 1e6
                        for _ in range(100):
                            M = (L + R) / 2
                            cdf = chi2comb_cdf(M**2, chi2s, 0.0)[0]
                            if cdf >= self.quantile:
                                R = M
                            else:
                                L = M
                        score = R
                    elif self.statistics == 'mean':
                        mean = 0
                        for c_i, r_i in zip(center_rotated, ellipsoid_radii):
                            mean += r_i * (c_i ** 2 + 1)
                        score = mean
                    elif self.statistics == 'mean-and-variance':
                        mean = 0
                        var = 0
                        for c_i, r_i in zip(center_rotated, ellipsoid_radii):
                            mean += r_i * (c_i ** 2 + 1)
                            var += r_i**2 * (2 * (1 + 2 * c_i**2))
                        score = [mean, var]
                    else:
                        raise ValueError('statistics should be either cdf, mean or mean-and-variance')
            
            all_scores.append(score)
            
            
        
        for param in likelihood_model.parameters():
            # turn on requires_grad
            param.requires_grad_()
            
            
        return np.array(all_scores)
    
class LatentScore(OODBaseMethod):
    """
    Calculates a score only based on the latent representation of the data.
    
    For example, if we are considering an Lp norm based method and p is given and the
    score is the norm, then it just calculates the norm of the latent representation 
    of the data. According to Nalisnick et al. (2019), the norm for p=2 of the latent 
    representation should fix the pathologies 
    
    TODO: for some reason we can't reproduce for CIFAR10-vs-SVHN
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
        #
        score_cls: th.Optional[str] = None,
        score_args: th.Optional[th.Dict[str, th.Any]] = None,
        tunable_parameter_name: th.Optional[str] = None,
        tunable_lr: th.Optional[th.Tuple[float, float]] = None,
        show_std: bool = False,
        # 
        progress_bar: bool = True,
        bincount: int = 5,
        
        visualize_reference: bool = False,
        
        **kwargs,
    ) -> None:
        super().__init__(
            x_loader=x_loader, 
            x=x, 
            x_batch=x_batch, 
            likelihood_model=likelihood_model, 
            logger=logger, 
            in_distr_loader=in_distr_loader, 
            **kwargs
        )
        
        if x is not None:
            self.x_batch = x.unsqueeze(0)
        
        self.progress_bar = progress_bar
        
        self.bincount = bincount
        
        score_args = score_args if score_args is not None else {}

        if score_cls is None:
            raise ValueError("score_cls must be provided!")
        
        self.score_module = dy.eval(score_cls)(**score_args, 
                                               in_distr_loader=in_distr_loader, 
                                               likelihood_model=likelihood_model)
        
        self.has_tunable_parameter = False
        if tunable_parameter_name is not None:
            self.tunable_parameter_name = tunable_parameter_name
            self.parameter_values = np.linspace(*tunable_lr, self.bincount)
            self.has_tunable_parameter = True
            self.show_std = show_std
        
        # TODO: Change this to a random thing dependant on a seed
        self.in_distr_batch, _, _ = next(iter(in_distr_loader))
        
        self.visualize_reference = visualize_reference
        
            
    def run(self):
        """
        Creates a histogram of scores, with the scores being the lp-norm of the latent representation of the data.
        """
        if not hasattr(self.likelihood_model, '_nflow'):
            raise ValueError('The likelihood model must have a _nflow attribute that returns the number of flows.')
        
        kwargs = {}
        if self.has_tunable_parameter:
            kwargs[self.tunable_parameter_name] = self.parameter_values
        
        def get_scores(x_batch):
            with torch.no_grad():
                z = self.likelihood_model._nflow.transform_to_noise(x_batch)  
            return self.score_module(z, x_batch, self.likelihood_model, **kwargs)
        
        if self.x_batch is None:
            raise ValueError('x_batch must be provided!')
            # TODO: somewhow incorporate the x_loader as well
            # if self.progress_bar:
            #     iterable = tqdm(self.x_loader)
            # else:
            #     iterable = self.x_loader
            # for x_batch, _ in iterable:
            #     with torch.no_grad():
            #         z = self.likelihood_model._nflow.transform_to_noise(x_batch)  
                
            #     new_scores = self.score_module(z, x_batch, self.likelihood_model, **kwargs) 
            #     all_scores = np.concatenate([all_scores, new_scores], dim=0) if all_scores is not None else new_scores
        else:
            all_scores = get_scores(self.x_batch)

        def visualize_scores(all_scores, reference_scores = None):
            """
            This function visualizes scores.
            
            If all_scores is one dimensional, it will create a histogram of the scores.
            If it is nx2 dimensional, it will create a scatter plot of the scores.
            If it is nxr it will create a line plot of the scores across all the r dimensions.
            """
            if len(all_scores.shape) > 1 and all_scores.shape[1] > 2:
                mean_scores = []
                mean_minus_std = []
                mean_plus_std = []
                mean_reference_scores = []
                
                for r, i in zip(self.parameter_values, range(all_scores.shape[1])):
                    scores = all_scores[:, i]
                    avg_scores = np.mean(scores)
                    upper_scores = scores[scores >= avg_scores]
                    lower_scores = scores[scores <= avg_scores]
                    
                    mean_scores.append(avg_scores)
                    mean_minus_std.append(avg_scores - np.std(lower_scores))
                    mean_plus_std.append(avg_scores + np.std(upper_scores))
                    
                    if reference_scores is not None:
                        reference_scores_ = reference_scores[:, i]
                        mean_reference_scores.append(np.mean(reference_scores_))
                
                if self.show_std:
                    wandb.log({
                        f"scoore-across-r": wandb.plot.line_series(
                            xs = self.parameter_values,
                            ys = [mean_scores, mean_minus_std, mean_plus_std],
                            keys = ["mean", "mean-std", "mean+std"],
                            title = "Scores",
                            xname = self.tunable_parameter_name,
                        )
                    })
                else:
                    ys = [mean_scores]
                    keys = ["mean_scores"]
                    if reference_scores is not None:
                        ys = [mean_reference_scores] + ys
                        keys = ["reference_scores", "mean_scores"]
                    wandb.log({
                        f"score-across-r": wandb.plot.line_series(
                            xs = self.parameter_values,
                            ys = ys,
                            keys = keys,
                            title = "Scores",
                            xname = self.tunable_parameter_name,
                        )
                    })
            elif len(all_scores.shape) > 1 and all_scores.shape[1] == 2:
                data = []
                for i in range(all_scores.shape[0]):
                    x, y = all_scores[i, 0], all_scores[i, 1]
                    data.append([x, y])
                table = wandb.Table(data=data, columns = ["mean", "variance"])
                wandb.log({f"score-scatter": wandb.plot.scatter(table, "mean", "variance", title="First and second order statistics")})
            else:    
                # sort all_scores 
                all_scores = np.sort(all_scores)
                
                # L = int(0.1 * len(all_scores))
                # R = int(0.9 * len(all_scores))
                # all_scores = all_scores[L: R]
                
                # create a density histogram out of all_scores
                # and store it as a line plot in (x_axis, density)
                hist, bin_edges = np.histogram(all_scores, bins=self.bincount, density=True)
                density = hist / np.sum(hist)
                centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                # get the average distance between two consecutive centers
                avg_dist = np.mean(np.diff(centers))
                # add two points to the left and right of the histogram
                # to make sure that the plot is not cut off
                centers = np.concatenate([[centers[0] - avg_dist], centers, [centers[-1] + avg_dist]])
                density = np.concatenate([[0], density, [0]])
                
                data = [[x, y] for x, y in zip(centers, density)]
                table = wandb.Table(data=data, columns = ['score', 'density'])
                wandb.log({f"score-density": wandb.plot.line(table, 'score', 'density', title='Score density')})
        
        if self.visualize_reference:
            reference_scores = get_scores(self.in_distr_batch)
            visualize_scores(all_scores, reference_scores)
        else:
            visualize_scores(all_scores)