"""
The main file used for OOD detection.
"""
import sys
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
import os
from model_zoo.datasets import get_loaders
import yaml
import traceback
import typing as th
from model_zoo.utils import load_model_with_checkpoints

@dataclass
class OODConfig:
    base_model: dict
    data: dict
    ood: dict
    logger: dict


def plot_likelihood_ood_histogram(
    model: torch.nn.Module,
    data_loader_in: torch.utils.data.DataLoader,
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
                            Defaults to 1000.
    """
    # create a function that returns a list of all the likelihoods when given
    # a dataloader
    def list_all_scores(dloader: torch.utils.data.DataLoader):
        log_probs = []
        for tmp in dloader:
            x = tmp
            # print("-------")
            # print(x.device)
            # print(model.device)
            
            t = model.log_prob(x).cpu().detach()
            # turn t into a list of floats
            t = t.flatten()
            t = t.tolist()
            log_probs += t
            if limit is not None and len(log_probs) > limit:
                break
        return log_probs

    # List the likelihoods for both dataloaders
    in_distr_scores = list_all_scores(data_loader_in)
    out_distr_scores = list_all_scores(data_loader_out)
    
    # plot using matplotlib and then visualize the picture 
    # using W&B media.
    try:
        # return an image of the histogram
        plt.hist(in_distr_scores, density=True, bins=100,
                 alpha=0.5, label="in distribution")
        plt.hist(out_distr_scores, density=True, bins=100,
                 alpha=0.5, label="out distribution")
        plt.title("Histogram of log likelihoods")
        plt.legend(loc="upper right")
        buf = io.BytesIO()
        # Save your plot to the buffer
        plt.savefig(buf, format="png")

        # Use PIL to convert the BytesIO object to an image object
        buf.seek(0)
        img = Image.open(buf)
    finally:
        plt.close()

    return np.array(img)


def run_ood(config: dict):
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
    
    model = load_model_with_checkpoints(config=config['base_model'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    ##################
    # (1) Data setup #
    ##################
    in_train_loader, _, in_test_loader = get_loaders(
        **config["data"]["in_distribution"]["dataloader_args"],
        device=device,
        shuffle=False,
        data_root='data/',
        unsupervised=True,
    )
    ood_train_loader, _, ood_test_loader = get_loaders(
        **config["data"]["out_of_distribution"]["dataloader_args"],
        device=device,
        shuffle=False,
        data_root='data/',
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
    else:
        raise ValueError("pick_loader should be either test or train")
    
    # out_loader is the loader that is used for the out-of-distribution data
    if not 'pick_loader' in config['data']['out_of_distribution']:
        print("pick_loader for ood not in config, setting to test")
        config['data']['out_of_distribution']['pick_loader'] = 'test'
        
    if config['data']['out_of_distribution']['pick_loader'] == 'test':
        out_loader = ood_test_loader
    elif config['data']['out_of_distribution']['pick_loader'] == 'train':
        out_loader = ood_train_loader
    else:
        raise ValueError("pick_loader should be either test or train")


    ############################################################
    # (3) Log model samples and in/out of distribution samples #
    ############################################################
    
    # print out a sample ood and in distribution image onto the wandb logger
    np.random.seed(config["data"]["seed"])

    # you can set to visualize or bypass the visualization for speedup!
    if 'bypass_visualization' not in config['ood'] or not config['ood']['bypass_visualization']:
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
        
        # generate 9 samples from the model if bypass sampling is not set to True
        if 'bypass_samples_visualization' not in config['ood'] or not config['ood']['bypass_samples_visualization']:
            with torch.no_grad():
                # set torch seed for reproducibility
                if config["ood"]["seed"] is not None:
                    torch.manual_seed(config["ood"]["seed"])
                samples = model.sample(9)
                samples = torchvision.utils.make_grid(samples, nrow=3)
                wandb.log(
                    {"data/model_generated": [wandb.Image(samples, caption="model generated")]})
            
        img_array = plot_likelihood_ood_histogram(
            model,
            in_loader,
            out_loader,
        )
        wandb.log({"likelihood_ood_histogram": [wandb.Image(
            img_array, caption="Histogram of log likelihoods")]})
    
    #########################################
    # (4) Instantiate an OOD solver and run #
    #########################################
    
    # For dummy runs that you just use for visualization
    if "method_args" not in config["ood"] or "method" not in config["ood"]:
        print("No ood method available! Exiting...")
        return
    
    method_args = copy.deepcopy(config["ood"]["method_args"])
    method_args["logger"] = wandb.run
    method_args["likelihood_model"] = model

    # pick a random batch with seed for reproducibility
    if config["ood"]["seed"] is not None:
        np.random.seed(config["ood"]["seed"])
    idx = np.random.randint(len(out_loader))
    for _ in range(idx + 1):
        x = next(iter(out_loader))

    if config["ood"]["pick_single"]:
        # pick a single image the selected batch
        method_args["x"] = x[np.random.randint(x.shape[0])]
    elif "use_dataloader" in config["ood"] and config["ood"]["use_dataloader"]:
        method_args["x_loader"] = out_loader
    elif "pick_count" not in config["ood"]:
        raise ValueError("pick_count not in config when pick_single=False")
    else:
        # pass in the entire batch
        r = min(config["ood"]["pick_count"], x.shape[0])
        method_args["x_batch"] = x[:r]
    
    method_args["in_distr_loader"] = in_train_loader
    
    torch.manual_seed(110)
    method = dy.eval(config["ood"]["method"])(**method_args)

    # Call the run function of the given method
    method.run()

def dysweep_run(config, checkpoint_dir):
    """
    Function compatible with dysweep
    """
    try:
        run_ood(config)
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
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to the config file",
    )
    args = parser.parse_args()
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

    run_ood(conf)
