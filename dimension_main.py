#!/usr/bin/env python3

import random
import argparse
import pprint
import torch
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict, fields
# from config import get_dimension_config, parse_config_arg
# from two_step_zoo import get_writer
from dimensions.datasets import load_data
import typing as th
from jsonargparse import ArgumentParser, ActionConfigFile
from random_word import RandomWords
import wandb
from dotenv import load_dotenv
import os
from dysweep import parse_dictionary_onto_dataclass
from model_zoo.datasets.loaders import get_loaders
import dypy as dy
from model_zoo import (
   get_evaluator, Writer
)

@dataclass
class EstimatorConfig:
    method: str
    method_args: th.Optional[dict] = None
    fitting_args: th.Optional[dict] = None

@dataclass
class WriterConfig:
    tag_group: str
    type: str
    entity: str
    project: str
    project: str
    name: th.Optional[str] = None
    make_subdir: bool = True
    explicit_subdir: th.Optional[str] = None
    logdir: th.Optional[str] = None
    
@dataclass
class IntrinsicDimensionConfig:
    intrinsic_dimension_estimator: EstimatorConfig
    writer: WriterConfig
    seed: th.Optional[int] = None
    data: th.Optional[th.Dict[str, th.Any]] = None
    local: bool = True

def run(
    args: IntrinsicDimensionConfig,
    checkpoint_dir: th.Optional[str] = None,
    gpu_index: int = -1,
):
    
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
        # Set it to unsupervised to omit the labels
        train_ready=True,
        **args.data,
    )
    
    # Create a writer according to the input configuration
    # if the checkpoint directory is explicitly given as an input to the function, then the logdir would be equal to the checkpoint_dir
    # otherwise, it would either be explicitly given in the configuration, or it would be set according to the environment variable MODEL_DIR
    writer_args = args.writer.as_dict()
    if checkpoint_dir is not None:
        writer = Writer(logdir=checkpoint_dir, make_subdir=False, **writer_args)
    else:
        if writer_args['logdir'] is None:
            writer_args['logdir'] = os.environ.get('MODEL_DIR', './runs')
        writer = Writer(**writer_args)
    
    intrinsic_dimension_estimator = dy.eval(args.intrinsic_dimension_estimator.method)(
        train_dataloader = train_loader,
        valid_dataloader = valid_loader,
        test_dataloader = test_loader,
        writer=writer,
        device=device,
        **(args.intrinsic_dimension_estimator.method_args or {})
    )
    
    intrinsic_dimension_estimator.fit(
        **(args.intrinsic_dimension_estimator.fitting_args or {})
    )
    
    if args.local:
        
        table = []
        for x, y, idx in test_loader:
            estimated_lid = intrinsic_dimension_estimator.estimate_lid(x)
            for lid_, y_, idx_ in zip(estimated_lid, y, idx):
                table.append([lid_, y_, idx_])
                
        # TODO: this should go through writer instead of weights and biases
        # extend this to other runnable modules as well!
        writer.write_table(
            name='LID_data',
            data=table,
            columns = ['predicted_lid', 'true_lid', 'idx'],
        )
        
        writer.log_scatterplot(
            name='LID_scatterplot',
            title="LID estimation scatterplot",
            data_table_ref='LID_data',
            x='predicted_lid',
            y='true_lid',
        )
        
    else:
        id_est = intrinsic_dimension_estimator.estimate_id()

        print("Estimated global intrinsic dimension (ID): ", id_est)

def dysweep_compatible_run(config, checkpoint_dir, gpu_index: int = -1):
    args = parse_dictionary_onto_dataclass(config, IntrinsicDimensionConfig)
    run(args, checkpoint_dir, gpu_index)
    
if __name__ == "__main__":
    # create a jsonargparse that gets a config file
    parser = ArgumentParser()
    parser.add_class_arguments(
        IntrinsicDimensionConfig,
        fail_untyped=False,
        sub_configs=True,
    )
    # add an argument to the parser pertaining to the gpu index
    parser.add_argument(
        '--gpu-index',
        type=int,
        help="The index of GPU being used",
        default=0,
    )
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to the config file",
    )
    
    
    args = parser.parse_args()
    
    
    print("Running on gpu index", args.gpu_index)
    
    args.writer.name = f"{args.writer.name or ''}-{RandomWords().get_random_word()}"

    run(args, gpu_index=args.gpu_index)

