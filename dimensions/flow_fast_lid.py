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

class FastFlowLID(BaseIntrinsicDimensionEstimationMethod):
    def __init__(
        self,
        *args,
        flow_model: th.Optional[NormalizingFlow] = None,
        flow_model_class: th.Optional[str] = None,
        flow_model_args: th.Optional[dict] = None,
        checkpoint_tag: th.Optional[str] = 'latest',
        flow_model_trainer_args: th.Optional[dict] = None,
        verbose: int = 0,
        r: float = -20,
        submethod: th.Literal['rho_derivative', 'fast_regression', 'jacobian_threshold'] = 'rho_derivative',
        regression_points: th.Optional[th.List[float]] = None,
        **kwargs,
    ):
        # Call the parent class
        super().__init__(*args, **kwargs)
        
        if self.ambient_dim is None:
            raise ValueError("You should specify the ambient data dimensionality in the \'ambient_dim\' argument.")
        
        # Set the amount of verbosity
        self.verbose = verbose
        
        # The scale variable
        self.r = r
        
        # Added regression points if the method is regression-based
        self.regression_points = np.array(regression_points or [0, 0.1]) # TIP: small scale regression steps are always more accurate!
        self.submethod = submethod
        
        if flow_model is None:
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
        else:
            self.module = flow_model
            self.checkpoint_tag = None
            
    def setup_method(
        self,
        use_functorch: bool = True,
        use_forward_model: bool = True,
        use_vmap: bool = False,
        chunk_size: int = 1,
        diff_transform: bool = False,
        visualize_eigenspectrum: bool = False,
    ):
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
        if self.submethod == 'rho_derivative':
            self.rho_r_deriv_calculator = GaussianConvolutionRateStatsCalculator(
                likelihood_model = self.module,
                **encoding_model_args,
                # verbosity
                verbose = self.verbose,
                visualize_eigenspectrum=visualize_eigenspectrum,
            )
        elif self.submethod == 'fast_regression':
            self.rho_r_calculator = GaussianConvolutionStatsCalculator(
                likelihood_model = self.module,
                **encoding_model_args,
                # verbosity
                verbose = self.verbose,
                visualize_eigenspectrum=visualize_eigenspectrum,
            )
        elif self.submethod == 'jacobian_threshold':
            self.rho_r_deriv_calculator = JacobianThresholdStatsCalculator(
                likelihood_model = self.module,
                **encoding_model_args,
                # verbosity
                verbose = self.verbose,
                visualize_eigenspectrum=visualize_eigenspectrum,
            )
        else:
            raise ValueError(f"Unknown submethod {self.submethod}")
        
    def fit(
        self,
        use_functorch: bool = True,
        use_forward_model: bool = True,
        use_vmap: bool = False,
        chunk_size: int = 1,
        diff_transform: bool = False,
        visualize_data: bool = False,
        visualize_eigenspectrum: bool = False,
        bypass_training: bool = False,
        sample_count: int = 10,
    ):
        self.trainer.load_checkpoint('latest')
        # The actual training loop
        if not bypass_training:
            self.trainer.train()
        self.trainer.load_checkpoint(self.checkpoint_tag)
        
        if visualize_data:
            data_vis_img = print_data_2d(
                list_of_loaders=[self.train_dataloader, self.test_dataloader, [self.module.sample(10) for _ in range(sample_count)]],
                list_of_labels=['train_split', 'test_split', 'sample'],
                alphas=[0.2, 0.2, 0.7],
                batch_limit=10,
                close_matplotlib = True,
            )
            self.writer.write_image(
                tag='samples_2d',
                img_tensor=data_vis_img,
            )
        
        self.setup_method(
            use_functorch=use_functorch,
            use_forward_model=use_forward_model,
            use_vmap=use_vmap,
            chunk_size=chunk_size,
            diff_transform=diff_transform,
            visualize_eigenspectrum=visualize_eigenspectrum,
        )
        
    def _calc_lid(self, x, r: float, use_cache: bool = False, visualize_eigen_tag: str = ""):
        if self.submethod == 'fast_regression':
            first_time = not use_cache
            r_values = []
            rho_r_values = []
            for shft in self.regression_points:
                real_r = r + shft
                estimated_rho_r = self.rho_r_calculator.calculate_statistics(
                    r=real_r,
                    loader=[x],
                    use_cache=not first_time,
                    visualize_eigen_tag=visualize_eigen_tag,
                )
                r_values.append(real_r)
                rho_r_values.append(estimated_rho_r)
                first_time = False
            
            r_values = np.array(r_values).reshape((-1, 1))
            rho_r_values = np.array(rho_r_values)
            linear_model = LinearRegression()
            linear_model.fit(r_values, rho_r_values)
            
            return linear_model.coef_[0] + self.ambient_dim
        else:
            return self.rho_r_deriv_calculator.calculate_statistics(
                r = r,
                loader = [x],
                use_cache = use_cache,
                visualize_eigen_tag=visualize_eigen_tag,
            ) + (0 if self.submethod == 'jacobian_threshold' else self.ambient_dim)
    
    def estimate_lid(
        self, 
        x,
        show_trend: bool = False,
        left_r: float = -20.0, 
        right_r: float = 10.0,
        radii_n: int = 10,
        visualize_std: bool = False,
    ):
        first_time = True
        if show_trend:
            all_estimated_lid = []
            radii_range = np.linspace(left_r, right_r, radii_n)
            rng = radii_range
            if self.verbose > 0:
                rng = tqdm(rng, desc="Calculating for 'r'")
            for r in rng:
                all_estimated_lid.append(self._calc_lid(x, r, use_cache=not first_time))
                first_time = False
            visualize_trends(
                scores=np.stack(all_estimated_lid).T,
                t_values=radii_range,
                title="Estimated LID per r",
                x_label='r',
                y_label='LID_r',
                with_std=visualize_std,
            )
        return self._calc_lid(x, self.r, use_cache=not first_time)       
    
    def custom_run(
        self,
        batch_limit = 100,
        group_limit = 8,
        show_trend: bool = False,
        left_r: float = -20.0, 
        right_r: float = 10.0,
        radii_n: int = 10,
        visualize_std: bool = False,
        averaging_r: th.Optional[float] = None,
    ):
        idx = 0
        true_labels = []
        predicted_labels = []
        groups = {}
        
        for x, y, _ in self.test_dataloader:
            idx += 1
            if idx > batch_limit:
                break
            true_labels.append(y)
            predicted_labels.append(self._calc_lid(x, self.r))
            if show_trend:
                for x_, y_ in zip(x, y):
                    if y_.item() not in groups:
                        groups[y_.item()] = [x_]
                    elif len(groups[y_.item()]) < group_limit:
                        groups[y_.item()].append(x_)
        if show_trend:
            for y in groups.keys():
                first_time = True
                x = torch.stack(groups[y])
                all_estimated_lid = []
                radii_range = np.linspace(left_r, right_r, radii_n)
                rng = radii_range
                if self.verbose > 0:
                    rng = tqdm(rng, desc="Calculating for 'r'")
                for r in rng:
                    all_estimated_lid.append(self._calc_lid(x, r, use_cache=not first_time, visualize_eigen_tag=str(y)))
                    first_time = False
                visualize_trends(
                    scores=np.stack(all_estimated_lid).T,
                    t_values=radii_range,
                    title=f"Estimated LID per r for group {y}",
                    x_label='r',
                    y_label='LID_r',
                    with_std=visualize_std,
                )
        heatmap_image = print_heatmap(
            loader_data=self.test_dataloader,
            loader_estimate=predicted_labels,
            batch_limit=batch_limit,
            close_matplotlib = True,
            averaging_r = averaging_r,
        )
        self.writer.write_image(
            tag='LID_estimate_heatmap',
            img_tensor=heatmap_image,
        )
                