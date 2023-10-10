
# Configuration Guide

We have a systematic hierarchical configuration format that we use for all our experiments, we use this convention for better version control, transparency, and fast and extendable development. We use `jsonargparse` to define all your configurations in a `yaml` file. In the following, we will explain the details in the configuration files and how you can define your own configurations.

## Training Configurations

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

## OOD Detection Configurations

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