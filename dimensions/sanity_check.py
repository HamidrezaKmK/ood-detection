"""
The fast LID estimates obtained by linearly approximating the flow networks.
"""
import time
from dimensions.base_method import BaseIntrinsicDimensionEstimationMethod
import typing as th
import numpy as np
from tqdm import tqdm
import dypy as dy
from torch.utils.data import DataLoader
from model_zoo import get_evaluator
from model_zoo.density_estimator.flow import NormalizingFlow
from model_zoo.trainers.single_trainer import BaseTrainer
from ood.methods.intrinsic_dimension.latent_statistics import GaussianConvolutionRateStatsCalculator, GaussianConvolutionStatsCalculator, JacobianThresholdStatsCalculator
from visualization import print_data_2d, print_heatmap
from ood.visualization import visualize_trends
from sklearn.linear_model import LinearRegression
import torch
from scipy.stats import multivariate_normal
import wandb

class SanityCheckRhoR(BaseIntrinsicDimensionEstimationMethod):
    def __init__(
        self,
        *args,
        flow_model_class: th.Optional[str] = None,
        flow_model_args: th.Optional[dict] = None,
        checkpoint_tag: th.Optional[str] = 'latest',
        flow_model_trainer_args: th.Optional[dict] = None,
        verbose: int = 0,
        submethod: th.Literal['derivative', 'convolution'] = 'convolution',
        **kwargs,
    ):
        # Call the parent class
        super().__init__(*args, **kwargs)
        
        if self.ambient_dim is None:
            raise ValueError("You should specify the ambient data dimensionality in the \'ambient_dim\' argument.")
        
        # Set the amount of verbosity
        self.verbose = verbose
        
        self.module = dy.eval(flow_model_class)(**flow_model_args).to(self.device)
        if not isinstance(self.module, NormalizingFlow):
            raise Exception("The module is not a normalizing flow.")
        
        self.module.set_optimizer(
            flow_model_trainer_args['optimizer']
        )
            
        # Additional args used for trainer.
        additional_args = flow_model_trainer_args.get('additional_init_args', None) or {}
        
        # Set an evaluator that gets called after each epoch of the trainer
        # for potential early stopping or evaluation
        evaluator = get_evaluator(
            self.module,
            train_loader=self.train_dataloader, valid_loader=self.valid_dataloader, test_loader=self.test_dataloader,
            valid_metrics=flow_model_trainer_args['evaluator']["valid_metrics"],
            test_metrics=flow_model_trainer_args['evaluator']["test_metrics"],
            **flow_model_trainer_args['evaluator'].get("metric_kwargs", {}),
        )
    
        self.trainer: BaseTrainer = dy.eval(flow_model_trainer_args['trainer_cls'])(
            self.module,
            ckpt_prefix="de",   
            train_loader=self.train_dataloader,
            valid_loader=self.valid_dataloader,
            test_loader=self.test_dataloader,
            writer=self.writer,
            max_epochs=flow_model_trainer_args['max_epochs'],
            early_stopping_metric=flow_model_trainer_args['early_stopping_metric'],
            max_bad_valid_epochs=flow_model_trainer_args['max_bad_valid_epochs'],
            max_grad_norm=flow_model_trainer_args['max_grad_norm'],
            evaluator=evaluator,
            only_test=flow_model_trainer_args['only_test'],
            sample_freq=flow_model_trainer_args['sample_freq'],
            progress_bar=flow_model_trainer_args['progress_bar'],
            **additional_args
        )
        
        self.checkpoint_tag = checkpoint_tag

        self.submethod = submethod
    
    def fit(
        self,
        use_functorch: bool = True,
        use_forward_model: bool = True,
        use_vmap: bool = False,
        chunk_size: int = 1,
        diff_transform: bool = False,
        visualize_eigenspectrum: bool = False,
        bypass_training: bool = False,
        visualize_data: bool = True,
        sample_count: int = 100,
    ):
        self.trainer.load_checkpoint('latest')
        # The actual training loop
        if not bypass_training:
            self.trainer.train()
        self.trainer.load_checkpoint(self.checkpoint_tag)
        
        if visualize_data:
            data_vis_img = print_data_2d(
                list_of_loaders=[self.train_dataloader, [self.module.sample(10) for _ in range(sample_count)]],
                list_of_labels=['datapoints', 'model samples'],
                alphas=[0.2, 0.2],
                colors=["#1F78B4", "#dfc27d"],
                batch_limit=50,
                close_matplotlib = True,
            )
            self.writer.write_image(
                tag='samples_2d',
                img_tensor=data_vis_img,
            )
            
        encoding_model_args = dict(
            encoding_model_class= 'ood.intrinsic_dimension.EncodingFlow',
            encoding_model_args = {
                'use_functorch': use_functorch,
                'use_forward_mode': use_forward_model,
                'use_vmap': use_vmap,
                'chunk_size': chunk_size,
                'diff_transform': diff_transform,
                'verbose': self.verbose,
            },
        )
        
        if self.submethod == 'convolution':
            self.rho_r_calculator = GaussianConvolutionStatsCalculator(
                likelihood_model = self.module,
                **encoding_model_args,
                # verbosity
                verbose = self.verbose,
                visualize_eigenspectrum=visualize_eigenspectrum,
            )
        elif self.submethod == 'derivative':
            self.rho_r_calculator = GaussianConvolutionRateStatsCalculator(
                likelihood_model = self.module,
                **encoding_model_args,
                # verbosity
                verbose = self.verbose,
                visualize_eigenspectrum=visualize_eigenspectrum,
            )
        else:
            raise ValueError(f"Submethod {self.submethod} not supported!")

    
    def custom_run(
        self,
        left_r: float = -20.0, 
        right_r: float = 10.0,
        radii_n: int = 10,
    ):
        rng = np.linspace(left_r, right_r, radii_n)
        dset = self.test_dataloader.dataset.dset
        idx = 0
        for mode, covariance_matrix, lid, mixture_prob in zip(dset.modes, dset.covariance_matrices, dset.manifold_dims, dset.mixture_probs):
            idx += 1
            if self.verbose > 0:
                rng = tqdm(rng, desc=f"calculating trend for mode [{idx}/{len(dset.modes)}]")
            ground_truth = []
            estimated = []
            first = True
            for r in rng:
                D = covariance_matrix.shape[0]
                if self.submethod == 'convolution':
                    R = np.exp(r)
                    cov = covariance_matrix + R ** 2 * np.eye(covariance_matrix.shape[0])
                    real_statistics = - 0.5 * D * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(cov).logabsdet 
                    real_statistics += np.log(mixture_prob)
                elif self.submethod == 'derivative':
                    real_statistics = lid - D
                ground_truth.append(real_statistics)
                estimated.append(
                    self.rho_r_calculator.calculate_statistics(
                        r=r,
                        loader=[torch.from_numpy(mode).float().to(self.module.device).unsqueeze(0)],
                        use_cache=not first,
                    )[0]
                )
                first = False
            wandb.log({
                f"trend/comparison-mode-{idx}": wandb.plot.line_series(
                    xs = np.linspace(left_r, right_r, radii_n),
                    ys = [ground_truth, estimated],
                    keys = ['ground_truth', 'estimated'],
                    title = f'Estimated vs true for mode {idx} with LID={lid}',
                    xname = 'r',
                )
            })