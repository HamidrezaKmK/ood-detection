from model_zoo.density_estimator.diffusions import ScoreBasedDiffusion
from ood.base_method import OODBaseMethod
import typing as th
import torch
from ood.methods.utils import buffer_loader
import numpy as np
import os
import json
from ood.visualization import visualize_scatterplots
from tqdm import tqdm

class DiffusionIntrinsicDimensionOODDetection(OODBaseMethod):
    """
    LID based ood detection method for diffusion model
    """
    def __init__(
        self,
        
        # The basic parameters passed to any OODBaseMethod
        likelihood_model: ScoreBasedDiffusion,    
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        checkpoint_dir: th.Optional[str] = None,
        checkpointing_buffer: th.Optional[int] = None,
        
        # The intrinsic dimension calculator args
        intrinsic_dimension_calculator_args: th.Optional[th.Dict[str, th.Any]] = None,
        log_prob_kwargs: th.Optional[th.Dict[str, th.Any]] = None,
        # for logging args
        verbose: int = 0,
        
        # Hyper-parameters relating to the scale parameter that is being computed
        evaluate_r: float = -20,
        adaptive_measurement: bool = False,
        given_mean_lid: th.Optional[float] = None,
        adaptive_train_data_buffer_size: int = 15,    
        bin_search_iteration_count: int = 20,
    ):
        """
        Initializes the IntrinsicDimensionScore class for outlier detection using intrinsic dimensionality.
        arguments are:
        
        Args:
            likelihood_model: torch.nn.Module
                The likelihood model that is used to compute the likelihoods of the data points.
            x_loader: torch.utils.data.DataLoader
                The loader for the in-distribution data.
            in_distr_loader: torch.utils.data.DataLoader
                The loader for the in-distribution data.
            checkpoint_dir: str
                The directory where the checkpoints will be saved.
            checkpointing_buffer: int
                The size of the buffer that is used to checkpoint the data.
            intrinsic_dimension_calculator_args: dict
                The arguments that are passed to the intrinsic dimension calculator.
            verbose: int
                The verbosity level of the method.
            evaluate_r: float
                The value of the scale parameter that is being evaluated.
            adaptive_measurement: bool
                If set to True, the adaptive measure will set the scale according to the training data in a smart way
            given_mean_lid: float
                The mean LID that is given to the adaptive setting mode
            adaptive_train_data_buffer_size: int
                The size of the buffer that is used to adaptively set the scale
            bin_search_iteration_count: int
                The number of iterations that are used in the binary search to find the scale    
        """
        self.likelihood_model = likelihood_model
        super().__init__(
            x_loader=x_loader, 
            likelihood_model=likelihood_model, 
            in_distr_loader=in_distr_loader, 
            checkpoint_dir=checkpoint_dir,
        )
        self.checkpointing_buffer = checkpointing_buffer
        self.verbose = verbose
        self.evaluate_r = evaluate_r
        
        self.intrinsic_dimension_calculator_args = intrinsic_dimension_calculator_args or {}
        self.log_prob_kwargs = log_prob_kwargs or {}
        # When set to True, the adaptive measure will set the scale according to the training data in a smart way
        self.adaptive_measurement = adaptive_measurement
        self.given_mean_lid = given_mean_lid
        self.adaptive_train_data_buffer_size = adaptive_train_data_buffer_size
        self.bin_search_iteration_count = bin_search_iteration_count
    
    def _get_loader_dimensionality(
        self,
        r: float,
        loader,
    ):
        all_lids = []
        for x in loader:
            all_lids.append(
                self.likelihood_model.lid(
                    x,
                    r=r,
                    **self.intrinsic_dimension_calculator_args,
                )
            )
        all_lids = torch.cat(all_lids, dim=0)
        return all_lids.cpu().numpy()
        
    def run(self):
        
        # use training data to perform adaptive scaling of fast-LIDL
        # pick a subset of the training data loader for this adaptation
        buffer = buffer_loader(self.in_distr_loader, self.adaptive_train_data_buffer_size, limit=1)
        for _ in buffer:
            inner_loader = _
            break
        
        progress_dict = {}
        if self.checkpoint_dir is not None and os.path.exists(os.path.join(self.checkpoint_dir, 'progress.json')):
            with open(os.path.join(self.checkpoint_dir, 'progress.json'), 'r') as file:
                progress_dict = json.load(file)

        # Handle selecting the r either adaptive or otherwise      
        if 'evaluate_r' in progress_dict:
            self.evaluate_r = progress_dict['evaluate_r']
        else:
            if self.adaptive_measurement:
                        
                L = -50.0 
                R = 50.0
                
                if self.verbose > 0:
                    bin_search = tqdm(range(self.bin_search_iteration_count), desc="binary search to find the scale")
                else:
                    bin_search = range(self.bin_search_iteration_count)
                
                # perform a binary search to come up with a scale so that almost all the training data has dimensionality above
                # that threshold
                
                
                for ii in bin_search:
                    mid = (L + R) / 2
                    dimensionalities = self._get_loader_dimensionality(
                        r=mid,
                        loader=inner_loader,
                    )
                    if np.mean(dimensionalities) > self.given_mean_lid:
                        L = mid
                    else:
                        R = mid
                
                self.evaluate_r = L
                
            progress_dict['evaluate_r'] = self.evaluate_r
        
        print("Evaluating with scale:", self.evaluate_r)
        if self.checkpoint_dir is not None:
            with open(os.path.join(self.checkpoint_dir, 'progress.json'), 'w') as file:
                json.dump(progress_dict, file)
                
        if self.verbose > 0:
            print("running with scale:", self.evaluate_r)
        
        
        # All dimensionalities and all likelihoods update
        all_dimensionalities = None
        all_likelihoods = None
        if self.checkpoint_dir is not None:
            if os.path.exists(os.path.join(self.checkpoint_dir, 'all_likelihoods.npy')):
                all_likelihoods = np.load(os.path.join(self.checkpoint_dir, 'all_likelihoods.npy'), allow_pickle=True)
            if os.path.exists(os.path.join(self.checkpoint_dir, 'all_dimensionalities.npy')):
                all_dimensionalities = np.load(os.path.join(self.checkpoint_dir, 'all_dimensionalities.npy'), allow_pickle=True)
        
        idx = 0
        
        
        for inner_loader in buffer_loader(self.x_loader, self.checkpointing_buffer):
            idx += 1
            if 'buffer_progress' in progress_dict:
                if idx <= progress_dict['buffer_progress']:
                    continue
            if self.verbose > 0:
                print(f"Working with buffer [{idx}]")
            
            # compute and add likelihoods
            if self.verbose > 0:
                print("Computing likelihoods ... ")
                
            all_buffer_likelihoods = None
            inner_loader_rng = tqdm(inner_loader, total=len(inner_loader)) if self.verbose > 0 else inner_loader
            for x in inner_loader_rng:
                with torch.no_grad():
                    likelihoods = self.likelihood_model.log_prob(x, **self.log_prob_kwargs).cpu().numpy().flatten()
                    all_buffer_likelihoods = np.concatenate([all_buffer_likelihoods, likelihoods]) if all_buffer_likelihoods is not None else likelihoods
            
            if self.verbose > 0:
                print("Computing dimensionalities ... ")
                
            # compute dimensionalities
            all_buffer_dimensionalities = self._get_loader_dimensionality(
                r=self.evaluate_r,
                loader=inner_loader,
            ) 
            
            
            all_dimensionalities = np.concatenate([all_dimensionalities, all_buffer_dimensionalities]) if all_dimensionalities is not None else all_buffer_dimensionalities
            all_likelihoods = np.concatenate([all_likelihoods, all_buffer_likelihoods]) if all_likelihoods is not None else all_buffer_likelihoods
            
            progress_dict['buffer_progress'] = idx
            
            if self.checkpoint_dir is not None:
                np.save(os.path.join(self.checkpoint_dir, 'all_likelihoods.npy'), all_likelihoods)
                np.save(os.path.join(self.checkpoint_dir, 'all_dimensionalities.npy'), all_dimensionalities)
                
                with open(os.path.join(self.checkpoint_dir, 'progress.json'), 'w') as file:
                    json.dump(progress_dict, file)
            
            
            

        visualize_scatterplots(
            scores = np.stack([all_likelihoods, all_dimensionalities]).T,
            column_names=["log-likelihood", "LID"],
        )