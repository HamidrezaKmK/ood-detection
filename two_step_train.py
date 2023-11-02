"""
The main file used for training two step models.
This file is based on jsonargparse and can be run using the following scheme

python two_step_train.py -c config.yaml

"""
from dotenv import load_dotenv
import os
import torch
import typing as th
import jsonargparse
from jsonargparse import ActionConfigFile
import dypy as dy
from model_zoo import TwoStepComponent, TwoStepDensityEstimator
from model_zoo.generalized_autoencoder import GeneralizedAutoEncoder
from model_zoo.density_estimator import DensityEstimator
from model_zoo.trainers.single_trainer import BaseTrainer
from model_zoo.trainers.two_step_trainer import SequentialTrainer, BaseTwoStepTrainer
from model_zoo import (
   get_evaluator, Writer
)
from model_zoo.datasets.loaders import get_loaders
from dataclasses import dataclass
from pprint import pprint
from dysweep import parse_dictionary_onto_dataclass
import traceback


    
@dataclass
class TrainerConfig:
    trainer_cls: th.Optional[str] = None
    writer: th.Optional[dict] = None
    optimizer: th.Optional[dict] = None
    
    additional_init_args: th.Optional[dict] = None
    
    max_epochs: int = 100
    early_stopping_metric: th.Optional[str] = None
    max_bad_valid_epochs: th.Optional[int] = None
    max_grad_norm: th.Optional[float] = None
    only_test: bool = False
    sample_freq: th.Optional[int] = None
    progress_bar: bool = False
    load_best_valid_first: bool = False
    
@dataclass
class GAEConfig:
    model_cls: th.Optional[str] = None
    model_init_args: th.Optional[dict] = None
    trainer: th.Optional[TrainerConfig] = None
    evaluator: th.Optional[dict] = None

@dataclass
class DEConfig:
    model_cls: th.Optional[str] = None
    model_init_args: th.Optional[dict] = None
    trainer: th.Optional[TrainerConfig] = None
    evaluator: th.Optional[dict] = None

@dataclass
class SharedConfig:
    # add arguments in shared config for model itself
    model_init_args: th.Optional[dict] = None
    # shared evaluator
    evaluator: th.Optional[dict] = None
    # a trigger for whether we perform sequential training or not
    sequential_training: bool = False
    # the options for the shared trainer
    trainer: th.Optional[TrainerConfig] = None
    # unrelated stuff!
    load_pretrained_gae: bool = False
    freeze_pretrained_gae: bool = False
    pretrained_gae_path: th.Optional[str] = None
    
@dataclass
class TwoStepConfig:
    gae_config: th.Optional[GAEConfig] = None
    de_config: th.Optional[DEConfig] = None
    shared_config: th.Optional[th.Dict] = None
    data: th.Optional[th.Dict[str, th.Any]] = None
    
def run(args, checkpoint_dir=None, gpu_index: int = -1):
    
    # Load the environment variables
    load_dotenv()
    
    # Set the data directory if it is specified in the environment
    # variables, otherwise, set to './data'
    if 'DATA_DIR' in os.environ:
        data_root = os.environ['DATA_DIR']
    else:
        data_root = './data'
        
    # setup device if the GPU index is set in the environment
    if torch.cuda.is_available():
        if gpu_index == -1:
            device = "cuda"
        else:
            device = f"cuda:{gpu_index}"
    else:
        device = "cpu"
        
    # Get the loaders from the configuration
    train_loader, valid_loader, test_loader = get_loaders(
        device=device,
        data_root=data_root,
        train_ready=True,
        **args.data,
    )

    # Create the modules
    gae_module: GeneralizedAutoEncoder = dy.eval(args.gae_config.model_cls)(
        **args.gae_config.model_init_args, **(args.shared_config.model_init_args or {})
    ).to(device)
    
    de_module: DensityEstimator = dy.eval(args.de_module.model_cls)(
        **args.de_module.model_init_args, **(args.shared_config.model_init_args or {})
    ).to(device)
    
    # Set the optimizer for the modules
    gae_module.set_optimizer(args.gae_config.optimizer)
    de_module.set_optimizer(args.de_config.optimizer)
        
    two_step_module = TwoStepDensityEstimator(
        generalized_autoencoder=gae_module,
        density_estimator=de_module
    )

    # Set an evaluator for each model that gets called after each epoch of the trainer
    # for potential early stopping or evaluation
    gae_evaluator = get_evaluator(
        two_step_module.generalized_autoencoder,
        valid_loader=valid_loader, test_loader=test_loader,
        train_loader=train_loader,
        valid_metrics=args.gae_config.evaluator["valid_metrics"],
        test_metrics=args.gae_config.evaluator["test_metrics"],
        **args.gae_config.evaluator.get("metric_kwargs", {}),
    )
    de_evaluator = get_evaluator(
        two_step_module.density_estimator,
        valid_loader=None, test_loader=None, # Loaders must be updated later by the trainer
        train_loader=train_loader,
        valid_metrics=args.de_config.evaluator["valid_metrics"],
        test_metrics=args.de_config.evaluator["test_metrics"],
        **args.de_config.evaluator.get("metric_kwargs", {}),
    )
    shared_evaluator = get_evaluator(
        two_step_module,
        valid_loader=valid_loader, test_loader=test_loader, # Loaders must be updated later by the trainer
        train_loader=train_loader,
        valid_metrics=args.shared_config.evaluator["valid_metrics"],
        test_metrics=args.shared_config.evaluator["test_metrics"],
        **args.shared_config.evaluator.get("metric_kwargs", {}),
    )
    # create a writer with its logdir equal to the dysweep checkpoint_dir
    # if it is not None
    if checkpoint_dir is not None:
        writer = Writer(logdir=checkpoint_dir, make_subdir=False, **args.trainer.writer)
    else:
        writer = Writer(**args.trainer.writer)
    
    # Instantiate the GAE trainer
    additional_args = args.gae_config.trainer.additional_init_args or {}
    gae_trainer: BaseTrainer = dy.eval(args.gae_config.trainer.trainer_cls)(
        gae_module,
        ckpt_prefix="gae",   
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        writer=writer,
        max_epochs= args.gae_config.trainer.max_epochs,
        early_stopping_metric= args.gae_config.trainer.early_stopping_metric,
        max_bad_valid_epochs= args.gae_config.trainer.max_bad_valid_epochs,
        max_grad_norm=args.gae_config.trainer.max_grad_norm,
        evaluator=gae_evaluator,
        only_test=args.gae_config.trainer.only_test,
        sample_freq=args.gae_config.trainer.sample_freq,
        progress_bar=args.gae_config.trainer.progress_bar,
        **additional_args
    )

    # Instantiate the DE trainer
    additional_args = args.de_config.trainer.additional_init_args or {}
    de_trainer: BaseTrainer = dy.eval(args.de_config.trainer.trainer_cls)(
        de_module,
        ckpt_prefix="de",   
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        writer=writer,
        max_epochs= args.de_config.trainer.max_epochs,
        early_stopping_metric= args.de_config.trainer.early_stopping_metric,
        max_bad_valid_epochs= args.de_config.trainer.max_bad_valid_epochs,
        max_grad_norm=args.de_config.trainer.max_grad_norm,
        evaluator=de_evaluator,
        only_test=args.de_config.trainer.only_test,
        sample_freq=args.de_config.trainer.sample_freq,
        progress_bar=args.de_config.trainer.progress_bar,
        **additional_args
    )
    
    if args.shared_config.sequential_training:
        additional_args = args.shared_config.trainer.additional_init_args or {}
        trainer: SequentialTrainer = dy.eval(args.shared_config.trainer.trainer_cls)(
            gae_trainer=gae_trainer,
            de_trainer=de_trainer,
            writer=writer,
            evaluator=shared_evaluator,
            checkpoint_load_list = ["best_valid", "latest"] if args.shared_config.trainer.load_best_valid_first else ["latest", "best_valid"]
            pretrained_gae_path=args.shared_config.pretrained_gae_path,
            freeze_pretrained_gae=args.shared_config.freeze_pretrained_gae,
            only_test=args.shared_config.trainer.only_test,
            **additional_args
        )
    else:
        
        # Additional args used for trainer.
        additional_args = args.shared_config.trainer.additional_init_args or {}
        trainer: BaseTwoStepTrainer = dy.eval(args.shared_config.trainer.trainer_cls)(
            two_step_module=two_step_module,
            gae_trainer=gae_trainer,
            de_trainer=de_trainer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            writer=writer,
            max_epochs=args.shared_config.trainer.max_epochs,
            early_stopping_metric=args.shared_config.trainer.early_stopping_metric,
            max_bad_valid_epochs=args.shared_config.trainer.max_bad_valid_epochs,
            max_grad_norm=args.shared_config.trainer.max_grad_norm,
            evaluator=shared_evaluator,
            checkpoint_load_list=["best_valid", "latest"] if args.shared_config.trainer.load_best_valid_first else ["latest", "best_valid"],
            pretrained_gae_path=args.shared_config.pretrained_gae_path,
            freeze_pretrained_gae=args.shared_config.freeze_pretrained_gae,
            only_test=args.shared_config.trainer.only_test,
            **additional_args
        )

    # The actual training loop
    trainer.train()

def dysweep_compatible_run(config, checkpoint_dir, gpu_index: int = -1):
    args = parse_dictionary_onto_dataclass(config, TwoStepConfig)
    run(args, checkpoint_dir, gpu_index)

if __name__ == "__main__":
    # Setup a parser for the configurations according to the above dataclasses
    # we use jsonargparse to allow for nested configurations
    parser = jsonargparse.ArgumentParser(description="Single Density Estimation or Generalized Autoencoder Training Module")
    parser.add_class_arguments(
        TwoStepConfig,
        fail_untyped=False,
        sub_configs=True,
    )
    parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    # add an argument called gpu_core_index which is an integer defaulting to -1 in parser
    parser.add_argument("--gpu_core_index", type=int, default=-1, help="The gpu core to use when training on multiple gpus")
    args = parser.parse_args()
    run(args, gpu_index=args.gpu_core_index)