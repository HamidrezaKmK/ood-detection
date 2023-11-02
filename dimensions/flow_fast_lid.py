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
from ood.methods.intrinsic_dimension.latent_statistics import GaussianConvolutionRateStatsCalculator

class FastFlowLID(BaseIntrinsicDimensionEstimationMethod):
    def __init__(
        self,
        *args,
        flow_model_class: th.Optional[str] = None,
        flow_model_args: th.Optional[dict] = None,
        checkpoint_tag: th.Optional[str] = 'latest',
        flow_model_trainer_args: th.Optional[dict] = None,
        verbose: int = 0,
        r: float = -20,
        **kwargs,
    ):
        # Call the parent class
        super().__init__(*args, **kwargs)
        
        # Set the amount of verbosity
        self.verbose = verbose
        
        # The scale variable
        self.r = r
        
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

    
    def fit(
        self,
    ):
        self.trainer.load_checkpoint('latest')
        # The actual training loop
        self.trainer.train()
        self.trainer.load_checkpoint(self.checkpoint_tag)
        
        self.rho_r_deriv_calculator = GaussianConvolutionRateStatsCalculator(
            likelihood_model = self.module,
            encoding_model_class= 'ood.intrinsic_dimension.EncodingFlow',
            encoding_model_args = {
                'use_functorch': True,
                'use_forward_mode': True,
                'use_vmap': False,
                'chunk_size': 1,
                'diff_transform': False,
            },
            # verbosity
            verbose = self.verbose,
        )
        
        
    def estimate_lid(self, x):
        return self.rho_r_deriv_calculator.calculate_statistics(
            r=self.r,
            loader=[x],
        )