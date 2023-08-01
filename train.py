"""
The main file used for training models.
This file is based on jsonargparse and can be run using the following scheme

python train.py -c config.yaml

"""

import torch
import typing as th
import jsonargparse
from jsonargparse import ActionConfigFile
import dypy as dy
from model_zoo import TwoStepComponent
from model_zoo.trainers.single_trainer import BaseTrainer
from model_zoo import (
   get_evaluator, Writer
)
from model_zoo.datasets.loaders import get_loaders
from dataclasses import dataclass
from pprint import pprint
from dysweep import parse_dictionary_onto_dataclass

@dataclass
class ModelConfig:
    is_gae: bool = False
    class_path: th.Optional[str] = None
    init_args: th.Optional[dict] = None

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
    
    evaluator: th.Optional[dict] = None
    
@dataclass
class TrainingConfig:
    model: th.Optional[ModelConfig] = None
    trainer: th.Optional[TrainerConfig] = None
    data: th.Optional[th.Dict[str, th.Any]] = None

def run(args, checkpoint_dir=None):
    
    # setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the loaders from the configuration
    train_loader, valid_loader, test_loader = get_loaders(
        device=device,
        data_root='./data',
        train_ready=True,
        **args.data,
    )

    # Create the module 
    module: TwoStepComponent = dy.eval(args.model.class_path)(**args.model.init_args).to(device)
    # Set the appropriate optimizer
    module.set_optimizer(args.trainer.optimizer)

    # Set an evaluator that gets called after each epoch of the trainer
    # for potential early stopping or evaluation
    evaluator = get_evaluator(
        module,
        train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
        valid_metrics=args.trainer.evaluator["valid_metrics"],
        test_metrics=args.trainer.evaluator["test_metrics"],
        **args.trainer.evaluator.get("metric_kwargs", {}),
    )

    # create a writer with its logdir equal to the dysweep checkpoint_dir
    # if it is not None
    if checkpoint_dir is not None:
        writer = Writer(logdir=checkpoint_dir, make_subdir=False, **args.trainer.writer)
    else:
        writer = Writer(**args.trainer.writer)
        
    # Additional args used for trainer.
    additional_args = args.trainer.additional_init_args or {}
    trainer: BaseTrainer = dy.eval(args.trainer.trainer_cls)(
        module,
        ckpt_prefix="gae" if args.model.is_gae else "de",   
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        writer=writer,
        max_epochs=args.trainer.max_epochs,
        early_stopping_metric=args.trainer.early_stopping_metric,
        max_bad_valid_epochs=args.trainer.max_bad_valid_epochs,
        max_grad_norm=args.trainer.max_grad_norm,
        evaluator=evaluator,
        only_test=args.trainer.only_test,
        sample_freq=args.trainer.sample_freq,
        progress_bar=args.trainer.progress_bar,
        **additional_args
    )

    # The actual training loop
    trainer.train()

def dysweep_compatible_run(config, checkpoint_dir):
    args = parse_dictionary_onto_dataclass(config, TrainingConfig)
    run(args, checkpoint_dir)
    
if __name__ == "__main__":
    # Setup a parser for the configurations according to the above dataclasses
    # we use jsonargparse to allow for nested configurations
    parser = jsonargparse.ArgumentParser(description="Single Density Estimation or Generalized Autoencoder Training Module")
    parser.add_class_arguments(
        TrainingConfig,
        fail_untyped=False,
        sub_configs=True,
    )
    parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    args = parser.parse_args()
    run(args)
