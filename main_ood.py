
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
import pprint
import wandb
from dataclasses import dataclass
from random_word import RandomWords
import os
from model_zoo.evaluators.ood_helpers import plot_ood_histogram_from_run_dir
from config import load_config_from_run_dir
from model_zoo.datasets import get_loaders
from load_run import load_run
import yaml
import traceback

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
    limit: int = 1000,
):
    def list_all_scores(dloader: torch.utils.data.DataLoader):
        log_probs = []
        for tmp in dloader:
            x, y, _ = tmp
            t = model.log_prob(x).cpu().detach()
            # turn t into a list of floats
            t = t.flatten()
            t = t.tolist()
            log_probs += t
            if len(log_probs) > limit:
                break
        return log_probs

    in_distr_scores = list_all_scores(data_loader_in)
    out_distr_scores = list_all_scores(data_loader_out)
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
    # load all the dataset
    # NOTE: this is a janky way to do this and it is dependent on the place
    # where you run this script from
    model_conf_dir = config["base_model"]["config_dir"]
    # load the configuration file which is a yaml into a dictionary called model_conf
    with open(model_conf_dir, "r") as f:
        model_conf = yaml.load(f, Loader=yaml.FullLoader)
    
    if 'model' in model_conf:
        model_conf = model_conf['model']
    
    model = dy.eval(model_conf['class_path'])(**model_conf['init_args'])
    # load the checkpoint from the checkpoint_dir
    run_dir = config["base_model"]["run_dir"]
    load_dict = load_run(run_dir, module=model)
    model = load_dict["module"]
    # (1) Load all the datasets

    # load the original cfg that the model was trained with
    # and delete all the keys that are not related to the dataset
    cfg_original = load_config_from_run_dir(run_dir)
    delete_keys = []
    for key, val in cfg_original.items():
        if key not in [
            "dataset",
            "data_root",
            "make_valid_loader",
            "train_batch_size",
            "valid_batch_size",
            "test_batch_size",
        ]:
            delete_keys.append(key)
    for key in delete_keys:
        cfg_original.pop(key)

    # load the ood dataset
    for key, val in config["data"]["out_of_distribution"]["dataloader_args"].items():
        if key not in cfg_original:
            raise ValueError(f"key {key} not in cfg_original")
        cfg_original[key] = val
    ood_train_loader, _, ood_test_loader = get_loaders(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **cfg_original,
    )

    # load the in distribution dataset
    for key, val in config["data"]["in_distribution"]["dataloader_args"].items():
        if key not in cfg_original:
            raise ValueError(f"key {key} not in cfg_original")
        cfg_original[key] = val
    in_train_loader, _, in_test_loader = get_loaders(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **cfg_original,
    )

    # print out a sample ood and in distribution image onto the wandb logger
    np.random.seed(config["data"]["seed"])

    # get 9 random samples from the in distribution dataset
    in_samples = in_test_loader.dataset.x[np.random.randint(
        len(in_test_loader.dataset), size=9)]
    out_samples = ood_test_loader.dataset.x[np.random.randint(
        len(ood_test_loader.dataset), size=9)]
    in_samples = torchvision.utils.make_grid(in_samples, nrow=3)
    out_samples = torchvision.utils.make_grid(out_samples, nrow=3)
    wandb.log({"data/in_distribution_samples": [wandb.Image(
        in_samples, caption="in distribution_samples")]})
    wandb.log({"data/out_of_distribution samples": [wandb.Image(
        out_samples, caption="out of distribution samples")]})
    # generate 9 samples from the model
    model.eval()
    with torch.no_grad():
        # set torch seed for reproducibility
        if config["ood"]["seed"] is not None:
            torch.manual_seed(config["ood"]["seed"])
        samples = model.sample(9)
        samples = torchvision.utils.make_grid(samples, nrow=3)
        wandb.log(
            {"data/model_samples": [wandb.Image(samples, caption="model samples")]})

    img_array = plot_likelihood_ood_histogram(
        model,
        in_test_loader,
        ood_test_loader,
    )
    wandb.log({"likelihood_ood_histogram": [wandb.Image(
        img_array, caption="Histogram of log likelihoods")]})

    method_args = copy.deepcopy(config["ood"]["method_args"])
    method_args["logger"] = wandb.run
    method_args["likelihood_model"] = model

    # pick a random batch with seed for reproducibility
    if config["ood"]["seed"] is not None:
        np.random.seed(config["ood"]["seed"])
    idx = np.random.randint(len(ood_test_loader))
    for _ in range(idx):
        x, y, _ = next(iter(ood_test_loader))

    if config["ood"]["pick_single"]:
        # pick a single image the selected batch
        method_args["x"] = x[np.random.randint(x.shape[0])]
    elif "use_dataloader" in config["ood"] and config["ood"]["use_dataloader"]:
        method_args["x_loader"] = ood_test_loader
    elif "pick_count" not in config["ood"]:
        raise ValueError("pick_count not in config when pick_single=False")
    else:
        # pass in the entire batch
        r = min(config["ood"]["pick_count"], x.shape[0])
        method_args["x_batch"] = x[:r]
    
    method_args["in_distr_loader"] = in_test_loader
    
    method = dy.eval(config["ood"]["method"])(**method_args)

    # Call the run function of the given method
    method.run()

def dysweep_run(config, checkpoint_dir):
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
