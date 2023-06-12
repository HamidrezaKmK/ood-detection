#!/usr/bin/env python3

import pprint
import torch
import typing as th
import jsonargparse
from jsonargparse import ActionConfigFile
import dypy as dy

from config import get_single_config, load_config, parse_config_arg
from model_zoo import (
    get_single_module, get_single_trainer, get_loaders_from_config,
    get_writer, get_evaluator, get_ood_evaluator, Writer
)

from dataclasses import dataclass

@dataclass
class DataConfig:
    data_cls: th.Optional[str] = None
    data_args: th.Optional[dict] = None
    
@dataclass
class ModelConfig:
    class_path: th.Optional[str] = None
    init_args: th.Optional[dict] = None
    
@dataclass
class LegacyConfig:
    dataset: str
    model: th.Optional[str] = None
    is_gae: bool = False
    cfg: th.Optional[th.Dict[str, th.Any]] = None
    load_dir: str = ""
    test_ood: bool = False
    only_test: bool = False
    load_best_valid_first: bool = False

@dataclass
class TrainerConfig:
    writer: th.Optional[dict] = None
    
    # The frequency of sampling happening in the training loop.
    # if set to a number, then every `sample_freq` epochs, the model will be evaluated
    # visually by sampling a set of samples from the model.
    sample_freq: th.Optional[int] = None
    
    optimizer: th.Optional[dict] = None
    max_epochs: int = 100
    batch_size: int = 128
    num_workers: int = 4
    
@dataclass
class TrainingConfig:
    data: th.Optional[DataConfig] = None
    model: th.Optional[ModelConfig] = None
    trainer: th.Optional[TrainerConfig] = None
    legacy: th.Optional[LegacyConfig] = None
    
    
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


device = "cuda" if torch.cuda.is_available() else "cpu"

if args.legacy is not None:
    if args.legacy.load_dir:
        # NOTE: Not updating config values using cmd line arguments (besides max_epochs)
        #       when loading a run.
        cfg = load_config(
            args=args
        )
    else:
        cfg = get_single_config(
            dataset=args.legacy.dataset,
            model=args.legacy.model,
            gae=args.legacy.is_gae,
            standalone=True
        )
        if args.legacy.cfg is None:
            args.legacy.cfg = {}
        cfg = {**cfg, **{k: v for k, v in args.legacy.cfg.items()}}

    pprint.sorted = lambda x, key=None: x
    pp = pprint.PrettyPrinter(indent=4)
    print(10*"-" + "cfg" + 10*"-")
    pp.pprint(cfg)
else:
    raise NotImplementedError("Only legacy mode is supported for now, so please specify a legacy config.")

train_loader, valid_loader, test_loader = get_loaders_from_config(cfg, device)


if args.legacy is not None:
    if args.trainer is not None and args.trainer.writer is not None:
        additional_args = args.trainer.writer
    writer = get_writer(args.legacy, cfg=cfg, additional_writer_args=additional_args)

if args.model is not None:
    
    module = dy.eval(args.model.class_path)(**args.model.init_args).to(device)
    # if args.trainer.optimizer does not exist, then use the legacy cfg to initialize the optimizer
    # otherwise, use the optimizer configurations are specified in args.trainer.optimizer
    if args.trainer is None or args.trainer.optimizer is None:
        module.set_optimizer(cfg)
    else:  
        module.set_optimizer(args.trainer.optimizer)
else:
    module = get_single_module(
        cfg,
        data_dim=cfg["data_dim"],
        data_shape=cfg["data_shape"],
        train_dataset_size=cfg["train_dataset_size"]
    ).to(device)
    
if args.legacy.test_ood or "likelihood_ood_acc" in cfg["test_metrics"]:
    evaluator = get_ood_evaluator(
        module,
        device,
        cfg=cfg,
        include_low_dim=False,
        valid_loader=valid_loader,
        test_loader=test_loader,
        train_loader=train_loader,
        savedir=writer.logdir
    )
else:
    if cfg["early_stopping_metric"] == "fid" and "fid" not in cfg["valid_metrics"]:
        cfg["valid_metrics"].append("fid")

    evaluator = get_evaluator(
        module,
        train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
        valid_metrics=cfg["valid_metrics"],
        test_metrics=cfg["test_metrics"],
        **cfg.get("metric_kwargs", {}),
    )


additional_args = {}
if args.trainer is not None:
    additional_args['sample_freq'] = args.trainer.sample_freq
    
trainer = get_single_trainer(
    module=module,
    ckpt_prefix="gae" if cfg["gae"] else "de",
    writer=writer,
    cfg=cfg,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    evaluator=evaluator,
    only_test=args.legacy.only_test,
    **additional_args
)

checkpoint_load_list = ["latest", "best_valid"]
if args.legacy.load_best_valid_first: checkpoint_load_list = checkpoint_load_list[::-1]
for ckpt in checkpoint_load_list:
    try:
        trainer.load_checkpoint(ckpt)
        break
    except FileNotFoundError:
        print(f"Did not find {ckpt} checkpoint")

trainer.train()
