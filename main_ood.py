"""
The main file used for OOD detection that follows the dysweep conventions
"""
import copy
from PIL import Image
import io
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
import dypy as dy
from jsonargparse import ArgumentParser, ActionConfigFile
import wandb
from dataclasses import dataclass
from random_word import RandomWords
from model_zoo.datasets import get_loaders
import traceback
import typing as th
from model_zoo.utils import load_model_with_checkpoints
from dotenv import load_dotenv
import os
from tqdm import tqdm
from ood.visualization import visualize_histogram

@dataclass
class OODConfig:
    base_model: dict
    data: dict
    ood: dict
    logger: dict


def plot_likelihood_ood_histogram(
    model: torch.nn.Module,
    data_loader_out: torch.utils.data.DataLoader,
    limit: th.Optional[int] = None,
):
    """
    Run the model on the in-distribution and out-of-distribution data
    and then plot the histogram of the log likelihoods of the models to show
    the pathologies if it exists.
    
    Args:
        model (torch.nn.Module): The likelihood model that contains a log_prob method
        data_loader_in (torch.utils.data.DataLoader): A dataloader for the in-distribution data
        data_loader_out (torch.utils.data.DataLoader): A dataloader for the out-of-distribution data
        limit (int, optional): The limit of number of datapoints to consider for the histogram.
                            Defaults to None => no limit.
    """
    # create a function that returns a list of all the likelihoods when given
    # a dataloader
    model.eval()
    log_probs = []
    for x in tqdm(data_loader_out, desc=f"Calculating likelihoods"):
        with torch.no_grad():
            t = model.log_prob(x).cpu()
        # turn t into a list of floats
        t = t.flatten()
        t = t.tolist()
        log_probs += t
        if limit is not None and len(log_probs) > limit:
            break
    
    visualize_histogram(
        scores=np.array(log_probs),
        x_label="log_likelihoods",
        plot_using_lines=True,
        bincount=20,
        reject_outliers=0.1,
    )

def run_ood(config: dict, gpu_index: int = 0):
    """
    This function reads the OOD configurations
    (samples can be found in experiments/ood/single_runs)
    that contains different sections:
    config {
        'base_model': 
        <information about the base likelihood model such as initiating variables
        and checkpoints>
        'data':
        <information about the in-distribution and out-of-distribution datasets>
        'ood': {
            'method': <name of the method class to run, all the classes can be found in ood/methods>
            'method_args': <arguments to pass to the method class>
            
        }
        'logger': <information about the W&B logger in use>
    }
    
    What this function does is that it first creates the appropriate torch model.
    Then uses the in-distribution data and out-of-distribution data to create a histogram
    comparing the likelihood of the model on the in-distribution and out-of-distribution data.
    
    For sanity check, it also samples 9 data points from the model and 9 data points from the
    in and out of distribution datasets and logs them to the W&B logger.
    
    After that, according to a set of arguments, either
    1. A single data point is picked from out-of-distribution dataset and the method is run on it
    2. A single batch is picked from the out-of-distribution dataset and the method is run on it
    3. The entire out-of-distribution dataset is passed to the method
    
    For each setting the following configurations are appropriate:
    1. 'ood': {
        'seed': <set the seed for reproducibility>
        'pick_single': True,
    }
    2. 'ood': {
        'seed': <set the seed for reproducibility>
        'pick_single': False,
        'use_dataloader': False
        'pick_count': <the number of data points to pick from a batch>
    }
    3. 'ood': {
        'seed': <set the seed for reproducibility>
        'pick_single': False,
        'use_dataloader': True,
    }
    
    With either of these three setups the appropriate method which inherits from
    ood.methods.base.BaseOODMethod is passed in with (1) x (a single data point),
    (2) x_batch (a batch of data points) or (3) x_loader (a dataloader).   

    Args:
        config (dict): The configuration dictionary
    """
    ###################
    # (1) Model setup #
    ###################
    load_dotenv()
    
    if 'MODEL_DIR' in os.environ:
        model_root = os.environ['MODEL_DIR']
    else:
        model_root = './runs'

    device = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
    
    model = load_model_with_checkpoints(config=config['base_model'], root=model_root, device=device)
    
    
    model.to(device)
    
    ##################
    # (1) Data setup #
    ##################
    # Load the environment variables
    
    # Set the data directory if it is specified in the environment
    # variables, otherwise, set to './data'
    if 'DATA_DIR' in os.environ:
        data_root = os.environ['DATA_DIR']
    else:
        data_root = './data'
        
    in_train_loader, _, in_test_loader = get_loaders(
        **config["data"]["in_distribution"]["dataloader_args"],
        device=device,
        shuffle=False,
        data_root=data_root,
        unsupervised=True,
    )
    ood_train_loader, _, ood_test_loader = get_loaders(
        **config["data"]["out_of_distribution"]["dataloader_args"],
        device=device,
        shuffle=False,
        data_root=data_root,
        unsupervised=True,
    )
    
    
    # in_loader is the loader that is used for the in-distribution data
    if not 'pick_loader' in config['data']['in_distribution']:
        print("pick_loader for in-distribution not in config, setting to test")
        config['data']['in_distribution']['pick_loader'] = 'test'
    
    if config['data']['in_distribution']['pick_loader'] == 'test':
        in_loader = in_test_loader
    elif config['data']['in_distribution']['pick_loader'] == 'train':
        in_loader = in_train_loader
    
    # out_loader is the loader that is used for the out-of-distribution data
    if not 'pick_loader' in config['data']['out_of_distribution']:
        print("pick_loader for ood not in config, setting to test")
        config['data']['out_of_distribution']['pick_loader'] = 'test'
        
    if config['data']['out_of_distribution']['pick_loader'] == 'test':
        out_loader = ood_test_loader
    elif config['data']['out_of_distribution']['pick_loader'] == 'train':
        out_loader = ood_train_loader


    ############################################################
    # (3) Log model samples and in/out of distribution samples #
    ############################################################
    
    np.random.seed(config["data"].get("seed", None))
    ood_config = config.get("ood", {})
    # you can set to visualize or bypass the visualization for speedup!
    if not ood_config.get('bypass_visualization', False):
        # get 9 random samples from the in distribution dataset
        sample_set = np.random.randint(len(in_loader.dataset), size=9)
        in_samples = []
        for s in sample_set:
            in_samples.append(in_loader.dataset[s])
        sample_set = np.random.randint(len(out_loader.dataset), size=9)
        out_samples = []
        for s in sample_set:
            out_samples.append(out_loader.dataset[s])
        in_samples = torch.stack(in_samples)
        out_samples = torch.stack(out_samples)

        in_samples = torchvision.utils.make_grid(in_samples, nrow=3)
        out_samples = torchvision.utils.make_grid(out_samples, nrow=3)
        
        wandb.log({"data/in_distribution_samples": [wandb.Image(
            in_samples, caption="in distribution_samples")]})
        wandb.log({"data/out_of_distribution samples": [wandb.Image(
            out_samples, caption="out of distribution samples")]})
        
        # generate 16 samples from the model if bypass sampling is not set to True
        if 'samples_visualization' in ood_config:
            if ood_config['samples_visualization'] > 0:
                with torch.no_grad():
                    def log_samples():
                        samples = model.sample(16)
                        samples = torchvision.utils.make_grid(samples, nrow=4)
                        wandb.log(
                            {"data/model_generated": [wandb.Image(samples, caption="model generated")]})
                    # set torch seed for reproducibility
                    ood_seed = ood_config.get("seed", None)
                    if ood_seed is not None:
                        if device.startswith("cuda"):
                            torch.cuda.manual_seed(ood_seed)
                        torch.manual_seed(ood_seed)
                        log_samples()
                    else:
                        log_samples()
                        
            if ood_config['samples_visualization'] > 1:
                wandb.log({"data/most_probable": [wandb.Image(model.sample(-1).squeeze(), caption="max likelihood")]})
        
        def log_histograms():
            limit = ood_config.get('histogram_limit', None)
            plot_likelihood_ood_histogram(
                model,
                out_loader,
                limit=limit,
            )
            
        if "bypass_visualize_histogram" not in ood_config or not ood_config['bypass_visualize_histogram']:
            ood_seed = ood_config.get("seed", None)
            if ood_seed is not None:  
                if device.startswith("cuda"):
                    torch.cuda.manual_seed(ood_seed)
                    torch.manual_seed(ood_seed)
                log_histograms()
            else:
                log_histograms()
                
    
    #########################################
    # (4) Instantiate an OOD solver and run #
    #########################################
    
    # For dummy runs that you just use for visualization
    if "method_args" not in ood_config or "method" not in ood_config:
        print("No ood method available! Exiting...")
        return
    
    method_args = copy.deepcopy(ood_config["method_args"])
    method_args["likelihood_model"] = model

    # Pick a subset of the OOD dataloader for tractability if 
    # pick_count is specified
    ood_seed = ood_config.get("seed", None)
    np.random.seed(ood_seed)
        
    method_args["x_loader"] = out_loader
    t = min(ood_config.get("pick_count", len(out_loader)), len(out_loader))
    method_args["x_loader"] = []
    iterable_ = iter(out_loader)
    for _ in range(t):
        batch = next(iterable_)
        method_args["x_loader"].append(batch)
    method_args["in_distr_loader"] = in_train_loader
    
    if device.startswith("cuda"):
        torch.cuda.manual_seed(ood_seed)
    torch.manual_seed(ood_seed)
    method = dy.eval(ood_config["method"])(**method_args)
    # Call the run function of the given method
    method.run()

def dysweep_compatible_run(config, checkpoint_dir, gpu_index: int = 0):
    """
    Function compatible with dysweep
    """
    try:
        run_ood(config, gpu_index=gpu_index)
    except Exception as e:
        print("Exception:\n", e)
        print(traceback.format_exc())
        print("-----------")
        raise e

if __name__ == "__main__":
    # create a jsonargparse that gets a config file
    parser = ArgumentParser()
    parser.add_class_arguments(
        OODConfig,
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
    
    conf = {
        "base_model": args.base_model,
        "data": args.data,
        "ood": args.ood,
    }
    if "name" in args.logger:
        # add a random word to the name
        r = RandomWords()
        args.logger["name"] += f"-{r.get_random_word()}"

    wandb.init(config=conf, **args.logger)

    run_ood(conf, gpu_index=args.gpu_index)
