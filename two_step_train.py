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
    
    de_evaluator: th.Optional[dict] = None
    gae_evaluator: th.Optional[dict] = None
    
@dataclass
class GAEConfig:
    model_class: th.Optional[str] = None
    model_init_args: th.Optional[dict] = None
    trainer_config: th.Optional[TrainerConfig] = None
    
@dataclass
class DEConfig:
    model_class: th.Optional[str] = None
    model_init_args: th.Optional[dict] = None
    trainer_config: th.Optional[TrainerConfig] = None

@dataclass
class TwoStepConfig:
    gae: th.Optional[GAEConfig] = None
    de: th.Optional[DEConfig] = None
    shared: th.Optional[th.Dict] = None
    trainer: th.Optional[TrainerConfig] = None
    data: th.Optional[th.Dict[str, th.Any]] = None
    load_pretrained_gae: bool = False
    freeze_pretrained_gae: bool = False

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

    # Create the module 
    gae_module: GeneralizedAutoEncoder = dy.eval(args.gae_model.class_path)(
        **args.gae_model.init_args, **(args.shared_config or {})
    ).to(device)
    
    de_module: DensityEstimator = dy.eval(args.de_module.class_path)(
        **args.de_module.init_args, **(args.shared_config or {})
    ).to(device)
    
    two_step_module = TwoStepDensityEstimator(
        generalized_autoencoder=gae_module,
        density_estimator=de_module
    )
    
    # Set the appropriate optimizer
    two_step_module.set_optimizer(args.trainer.optimizer)

    # Set an evaluator that gets called after each epoch of the trainer
    # for potential early stopping or evaluation
    gae_evaluator = get_evaluator(
        two_step_module.generalized_autoencoder,
        valid_loader=valid_loader, test_loader=test_loader,
        train_loader=train_loader,
        valid_metrics=args.trainer.gae_evaluator["valid_metrics"],
        test_metrics=args.trainer.gae_evaluator["test_metrics"],
        **args.trainer.gae_evaluator.get("metric_kwargs", {}),
    )
    de_evaluator = get_evaluator(
        two_step_module.density_estimator,
        valid_loader=None, test_loader=None, # Loaders must be updated later by the trainer
        train_loader=train_loader,
        valid_metrics=args.trainer.de_evaluator["valid_metrics"],
        test_metrics=args.trainer.de_evaluator["test_metrics"],
        **args.trainer.de_evaluator.get("metric_kwargs", {}),
    )

    # create a writer with its logdir equal to the dysweep checkpoint_dir
    # if it is not None
    if checkpoint_dir is not None:
        writer = Writer(logdir=checkpoint_dir, make_subdir=False, **args.trainer.writer)
    else:
        writer = Writer(**args.trainer.writer)
        
    # Additional args used for trainer.
    additional_args = args.trainer.additional_init_args or {}
    return SingleTrainer(
        module=module,
        ckpt_prefix=ckpt_prefix,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        writer=writer,
        max_epochs=cfg["max_epochs"],
        early_stopping_metric=cfg["early_stopping_metric"],
        max_bad_valid_epochs=cfg["max_bad_valid_epochs"],
        max_grad_norm=cfg["max_grad_norm"],
        evaluator=evaluator,
        only_test=only_test,
        **kwargs,
    )
    gae_trainer = get_single_trainer(
        module=two_step_module.generalized_autoencoder,
        ckpt_prefix="gae",
        writer=writer,
        cfg=gae_cfg,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        evaluator=gae_evaluator,
        only_test=only_test,
    )
    de_trainer = get_single_trainer(
        module=two_step_module.density_estimator,
        ckpt_prefix="de",
        writer=writer,
        cfg=de_cfg,
        train_loader=None,
        valid_loader=None,
        test_loader=None,
        evaluator=de_evaluator,
        only_test=only_test,
    )

    checkpoint_load_list = ["best_valid", "latest"] if load_best_valid_first else ["latest", "best_valid"]

    if shared_cfg["sequential_training"]:
        return SequentialTrainer(
            gae_trainer=gae_trainer,
            de_trainer=de_trainer,
            writer=writer,
            evaluator=shared_evaluator,
            checkpoint_load_list=checkpoint_load_list,
            pretrained_gae_path=pretrained_gae_path,
            freeze_pretrained_gae=freeze_pretrained_gae,
            only_test=only_test
        )

    elif shared_cfg["alternate_by_epoch"]:
        trainer_class = AlternatingEpochTrainer
    else:
        trainer_class = AlternatingIterationTrainer

    return trainer_class(
        two_step_module=two_step_module,
        gae_trainer=gae_trainer,
        de_trainer=de_trainer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        writer=writer,
        max_epochs=shared_cfg["max_epochs"],
        early_stopping_metric=shared_cfg["early_stopping_metric"],
        max_bad_valid_epochs=shared_cfg["max_bad_valid_epochs"],
        max_grad_norm=shared_cfg["max_grad_norm"],
        evaluator=shared_evaluator,
        checkpoint_load_list=checkpoint_load_list,
        pretrained_gae_path=pretrained_gae_path,
        freeze_pretrained_gae=freeze_pretrained_gae,
        only_test=only_test
    )


    trainer: BaseTrainer = dy.eval(args.trainer.trainer_cls)(
        two_step_module=two_step_module,
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