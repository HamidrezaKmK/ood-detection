# OOD Detection for Likelihood-based Deep Generative Models

This repository explores different ways of performing OOD detection on likelihood-based deep generative models. 

## Problem Statement
Based on the paper ["Do deep generative models know what they don't know?"](https://openreview.net/pdf?id=H1xwNhCcYm), pradoxically, likelihood values alone are not a reliable indicator for whether a datapoint is OOD or not. However, these generative models are able to produce high-quality in-distribution data, therefore, their likelihood landscape (or their learned density) definitely contains the information required for OOD detection. In this repo, we consider different methods to leverage this information and improve OOD detection.
This codebase uses the generative models in [two_step_zoo](https://github.com/layer6ai/two_step_zoo), containing code from the paper ["Diagnosing and Fixing Manifold Overfitting in Deep Generative Models"](https://arxiv.org/abs/2204.07172) accepted to TMLR in July 2022. Here, the models are improved and the hyperparameters are tuned for this particular application.


## Setup

**Make sure that your python is `3.9` or higher; otherwise, some of the new autodiff functionalities we use might break. Also, you should install the nflows package from [here](https://github.com/HamidrezaKmK/nflows) which is a version of `nflows` that makes it functional for RQ-NSFs (it is already defined in the requirement files).**

For python environment, we support both `pip` and `conda`.

To install the requirements with pip run the following:

```bash
pip install -r requirements.txt
```

And for conda, you may run the following:

```bash
conda env create -f env.yml 
# This creates an environment called 'ood-detection' that you can activate
```

## Custom Environment Variables

You can set environment variables for the directories in which you store the checkpoints or datasets, otherwise, the code will automatically create a `runs` directory to store the model checkpoints and create a `data` directory to store dataset information.

```bash
# Set the directory where you store all the model checkpoints
dotenv set MODEL_DIR <root-path-to-model-configurations-and-weights>
# Set the directory where you store all the datasets
dotenv set DATA_DIR <root-path-to-data-directory>
```

## Running Single Experiments

The project is divided into two sections:

1. **Training models**: The codes for model training lie within the [model_zoo](./model_zoo/) directory. To run a specific model, define a training configuration and run `train.py` on that configuration. For example, to train a Neural Spline Flow on Fashion-MNIST, there is a training configuration defined at [train_config](./configurations/training/rq_nsf_fmnist.yaml). We use `jsonargparse` to define all your configurations in a `yaml` file, so you can run the following:

```bash
python train.py --config configurations/training/rq_nsf_fmnist.yaml
```

2. **Performing OOD-detection**: The codebase pertaining to OOD-detection lies within the [ood](./ood/) directory. Every OOD detection method is encapsulated within a class that inherits a base class defined in [OODMethodBaseClass](./ood/methods/base_method.py). To run experiments on OOD detection, one can pick any likelihood based model with specific checkpoints, specify an *in-distribution* dataset and an *out-of-distribution* dataset, and run the method. The `main_ood.py` is the runner script for this. Similar to the training configurations, we use `jsonargparse` to define all your configurations in a `yaml` file, so you can run the following example that performs a basic OOD detection technique on a Neural Spline Flow trained on Fashion-MNIST and then tests it on MNIST to see the pathology:

```bash
python main_ood.py --config configurations/ood/simple_rq_nsf_fmnist_mnist.yaml
```

For more information on how to define these configurations, please check out our comments in the `yaml` files that we have provided alongside our configuration [guide](./docs/configs.md).



## Weights and Biases Integration and Sweeps

We use [dysweep](https://github.com/HamidrezaKmK/dysweep), which is an integration with weights and biases for systematic experimentation (similar to Hydra). 
We have grouped our experiments into different `yaml` files containing all the hyperparameter setup necessary down to the detail. Each file contains an overview of a **group** of relevant experiments; this integration groups together our experiments and performs sweeps that allow for parallelism. For an overview, please refer to [meta configuration](./meta_configurations/).

### Setting up Weights and Biases

To run the experiments, we require you to create a Weights & Biases workplace and set up the login information according to the guidelines indicated [here](https://docs.wandb.ai/quickstart). In this workplace, our code will create a project named `final-report`, containing multiple sweeps. 

**Important note:** The current workspace in all the YAML files is set to `dgm-l6` in [meta configuration](./meta_configurations/), please change it to whatever workspace or entity you are working with.

### Running Sweeps

Weights & Biases creates sweep servers that help you to simultaneously run different experiments.
Every `yaml` file in [meta configuration](./meta_configurations/) contains information in a sweep that can be run using the following:
```bash
dysweep_create --config <path-to-meta-configuration>
```
For example, to run a sweep server that handles training all the greyscale images, you may run the following:
```bash
dysweep_create --config ./meta_configuration/training/grayscale_flows.yaml
```
After running each sweep, you will be given a sweep identifier from the sweep server which would in turn allow you to run the actual experiments in parallel. To initiate a process that takes an experiment from the sweep server and run it, you may run the following:
```bash
./meta_run_train.sh <sweep-id> # if the sweep is pertaining to a model training task
./meta_run_ood.sh <sweep-id> # if the sweep is pertaining to an OOD detection task
```

## Reproducing work based on this codebase

* For the local intrinsic dimension (LID) based method, please check [here](docs/reproduce_lid.md).