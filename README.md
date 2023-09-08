# OOD-detection using Peaks of the Density Function

This repository explores different ways of performing OOD detection on likelihood-based deep generative models. The codebase uses [two_step_zoo](https://github.com/layer6ai/two_step_zoo), containing code from the paper ["Diagnosing and Fixing Manifold Overfitting in Deep Generative Models"](https://arxiv.org/abs/2204.07172) accepted to TMLR in July 2022.


## Setup

**Make sure that your python is `3.9`**. Higher versions will conflict with the [chi2comb](https://pypi.org/project/chi2comb/) library for generalized chi-square computation (TODO: fix this dependency).

Install the requirements with pip:

```bash
pip install -r requirements.txt
```

## Custom Environment Variables

You should set your environment variables for the root directory of data and the root directory of models with the following:

```bash
dotenv set MODEL_DIR <root-path-to-model-configurations-and-weights>
dotenv set DATA_DIR <root-path-to-data-directory>
```

Otherwise, the code-base will create a `runs` and `data` directory in the root of the repository.

## Running Single Experiments

The project is divided into two sections:

1. **Training models**: The codebase pertaining to model training lie within the [model_zoo](./model_zoo/) directory. To run a specific model, define a training configuration and run `train.py` on that configuration. For example, to train a Neural Spline Flow on Fashion-MNIST, there is a training configuration defined at [train_config](./configurations/training/rq_nsf_fmnist.yaml). We use `jsonargparse` to define all your configurations in a `yaml` file, so you can run the following:

```bash
python train.py --config configurations/training/rq_nsf_fmnist.yaml
```

(TODO: add two step models and VAE implementation as well)

2. **Performing OOD-detection**: The codebase pertaining to OOD-detection lies within the [ood](./ood/) directory. Every OOD detection method is encapsulated within a class that inherits a base class defined in [OODMethodBaseClass](./ood/methods/base_method.py). To run experiments on OOD detection, one can pick any likelihood based model with specific checkpoints, specify an *in-distribution* dataset and an *out-of-distribution* dataset, and run the method. The `main_ood.py` is the runner script for this. Similar to the training configurations, we use `jsonargparse` to define all your configurations in a `yaml` file, so you can run the following example that performs a basic OOD detection technique on a Neural Spline Flow trained on Fashion-MNIST and then tests it on MNIST to see the pathology:

```bash
python main_ood.py --config configurations/ood/simple_rq_nsf_fmnist_mnist.yaml
```

For more information on how to define these configurations, please check out our comments in the `yaml` files that we have provided alongside our configuration [guide](./configurations/README.md).

## Performing Sweeps and Group Experiments

We use [dysweep](https://github.com/HamidrezaKmK/dysweep) for grouping together our experiments and performing sweeps. All the sweep configurations lie in [ood-meta](./meta_configurations/ood/) for OOD-detection-related configuration groups, and [training-meta](./meta_configurations/training/) for training-related configuration groups. To create a sweep, you can run the following:

```bash
dysweep_create --config sweep_configuration
```

For example, you can run a group experiment on all the different grayscale datasets containing Omniglot, MNIST, EMNIST, and Fashion-MNIST over two architectures of flow models: Neural Spline Flow and Glow. To do this, you can run the following:

```bash
dysweep_create --config meta_configurations/ood/grayscale_flows.yaml
```

After running this, you would be given a sweep identifier that you can use to perform training tasks on any device in parallel. An example command to run is the following:

```bash
dysweep_run_resume --package train --function dysweep_compatible_run --run_additional_args gpu_index:<device-index> --config meta_configurations/ood/grayscale_flows.yaml --sweep-id <sweep-id> --count <maximum-number-of-jobs-to-execute>
```

You can also run sweeps on OOD-detection-related tasks, the only difference here is that you would have to change the `--package` argument to `main_ood` instead of `train` and define a sweep configuration . To read more about how this format is curated, please refer to the [dysweep documentation](github.com/HamidrezaKmK/dysweep).

For ease of use, we've created the following runnable bash scripts that you can use to run the sweeps:

```bash
# chmod +x meta_run.sh
./meta_run.sh <package-name> <sweep-id> <maximum-number-of-jobs-to-execute (default = 1000)> <device-index (default = 0)> 
```

## Configuration Guide

We have a systematic hierarchical configuration format that we use for all our experiments, we use this convention for better version control, transparency, and fast and extendable development. We use `jsonargparse` to define all your configurations in a `yaml` file. In the following, we will explain the details in the configuration files and how you can define your own configurations.

### Training Configurations

```yaml
# (1)   The configurations pertaining to the dataset being used for training
data:
    # A list of all the datasets that are supported is given in:
    # model_zoo/datasets/utils.py
    dataset: <name-of-the-dataset>
    train_batch_size: <batch-size-for-training>
    test_batch_size: <batch-size-for-testing>
    valid_batch_size: <batch-size-for-validation>
    # Note that all the subconfigs will be passed directly to the
    # model_zoo/datasets/loaders.py and the get_loaders function
    # so feel free to define any extra arguments here

# (2)   The configurations pertaining to the model being used for training
#       This is typically a torch.nn.Module
model:
    # Whether the model is a generalized autoencoder or not
    is_gae: <bool>

    # The model class, all should inherit from the base class defined in
    # GeneralizedAutoEncoder or DensityEstimator
    # These models have certain properties such as a log_prob function
    # some optimizer defined, etc.

    class_path: <path-to-the-model-class>
    # Please refer to code documentation for the specified class to 
    # see the appropriate arguments
    init_args: <dictionary-of-the-arguments>

# (3)   The configurations pertaining to the optimizer being used for training
trainer:
    # The class of the trainer, an example is 
    # model_zoo.trainers.single_trainer.SingleTrainer
    # for training a single model, and for generalized autoencoders
    # different trainers might be taken into consideration
    # All of these classes should inherit from the base class defined in
    # *model_zoo.trainers.single_trainer.BaseTrainer*
    trainer_cls: <class-of-the-trainer>
    # configurations relating to the optimizer
    optimizer:
        # the class of the optimizer, an example is torch.optim.AdamW
        class_path: <torch-optimizer-class>
        # you can define the base lr here for example
        init_args:
            lr: <learning-rate>
            # additional init args for an optimizer

        # a scheduler used on top of the given optimizer
        lr_scheduler:
            # the class of the scheduler, an example is torch.optim.lr_scheduler.ExponentialLR
            class_path: <torch-scheduler-class>
            init_args: 
                gamma: <gamma>
                # additional init args for the scheduler

    writer: <dictionary-of-arguments>
    # all the arguments given to the writer class
    # being used. You can check the arguments
    # in the init arguments of the class
    # *model_zoo.writer.Writer*
    # NOTE: the wandb is only supported now
        
    evaluator:
        valid_metrics: [loss]
        test_metrics: [loss]

    sample_freq: <x> # logs the samples after every x epochs
    max_epochs: <x> # The number of epochs to run
    early_stopping_metric: loss # The metric used for early stopping on the validation
    max_bad_valid_epochs: <x> # The maximum number of bad validations needed for early stopping
    max_grad_norm: <x> # gradient clipping
    only_test: <bool> # TODO: ?
    progress_bar: <bool> # output a progress bar or not
```

### OOD Detection Configurations

```yaml
# (1)   The configurations pertaining to the likelihood model being used for OOD detection
base_model:
    # A directory containing the configuration used for instantiating the likelihood model itself
    # it can be a json or a yaml file
    config_dir: <path-to-json-or-yaml-file>
    # A directory containing the model weights of the likelihood model to work with a good fit one
    # This is produced by running the single step training models
    checkpoint_dir: <path-to-checkpoints>

# (2)   The configurations pertaining to the dataset pairs that are used for OOD detection
#       the first dataset is the in-distribution dataset and the second one is the out-of-distribution
data:
    # specify the datasets and dataloader configurations for the in and out of distribution data.
    in_distribution:
        pick_loader: <train/test/valid> # The loader to pick between the created loaders for OOD detection
        dataloader_args: <dataloader-args-for-in-distribution>
        # A dictionary defined here is directly passed to the get_loaders function in model_zoo/datasets/loaders.py

    out_of_distribution:
        pick_loader: <train/test/valid> # The loader to pick between the created loaders for in-distribution
        dataloader_args: <dataloader-args-for-in-distribution>
        # A dictionary defined here is directly passed to the get_loaders function in model_zoo/datasets/loaders.py
    
    # for reproducibility of the shuffling of the datasets
    seed: <x>

ood:
    # Visualization arguments:
    # By default, the OOD-detection method has rather heavy visualizations:
    #   1. It visualizes 9 samples in a grid from both in-distribution and out-of-distribution
    #   2. It visualizes 9 samples from the likelihood-based generative model itself
    #   3. It visualizes the histogram of the log-likelihoods of the in-distribution and out-of-distribution
    bypass_visualization: <bool> # (optional argument) if set to true, then all the 3 steps above are bypassed
    samples_visualization: <0,1,2> # (optional argument) it can be 0, 1, or 2, and it does the following:
    #  0:   bypasses the visualization of the samples from the 
    #       likelihood-based generative model (the second step above)
    #  1:   Only visualze the 9 samples generated
    #  2:   Visualize the most likely sample as well using the sample(-1) method.
    #       this would correspond to the 0-latent variables in flow models and VAE models
    bypass_visualize_histogram: <bool> # (optional argument) if set to true, then the histogram of the log-likelihoods is bypassed

    # Batch/Loader/Datapoint
    # OOD detection algorithms operate over either single datapoints, batches, or on entire loaders
    # Using the pick_single variable, you can specify whether you want to pick a single datapoint or not
    # if pick_single is set to false, you can specify whether you want to perform OOD detection on the entire
    # dataloader or not.
    # 
    pick_single: <bool>
    use_dataloader: <bool>
    pick_count: <x> # If used in the dataloader setting, it will pick at most 'x' batches from the dataloader
    # if used in the single datapoint setting, it will pick at most 'x' datapoints from the dataset

    # for reproducibility of the shuffling of the datasets
    seed: <x>
    
    # An ood method class that inherits from the base class defined in
    # ood.methods.base_method.OODMethodBaseClass
    method: <method-class>
    method_args: <dictionary-being-passed-for-initialization>
        

logger:
    project: <W&B-project>
    entity: <W&B-entity>
    name: <name-of-the-run>
```