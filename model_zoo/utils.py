import torch
from torch.utils.data import DataLoader
import functools
import dypy as dy
import typing as th
import yaml

def batch_or_dataloader(agg_func=torch.cat):
    def decorator(batch_fn):
        """
        Decorator for methods in which the first arg (after `self`) can either be
        a batch or a dataloader.

        The method should be coded for batch inputs. When called, the decorator will automatically
        determine whether the first input is a batch or dataloader and apply the method accordingly.
        """
        @functools.wraps(batch_fn)
        def batch_fn_wrapper(ref, batch_or_dataloader, **kwargs):
            if isinstance(batch_or_dataloader, DataLoader): # Input is a dataloader
                list_out = [batch_fn(ref, batch, **kwargs)
                            for batch, _, _ in batch_or_dataloader]

                if list_out and type(list_out[0]) in (list, tuple):
                    # Each member of list_out is a tuple/list; re-zip them and output a tuple
                    return tuple(agg_func(out) for out in zip(*list_out))
                else:
                    # Output is not a tuple
                    return agg_func(list_out)

            else: # Input is a batch
                return batch_fn(ref, batch_or_dataloader, **kwargs)

        return batch_fn_wrapper

    return decorator

def load_model_with_checkpoints(
    config,
    device: th.Optional[str] = None,
):
    model_conf = None
    # load the model and load the corresponding checkpoint
    if 'config_dir' in config:
        filename = config['config_dir']
        with open(filename, 'r') as f:
            model_conf = yaml.load(f, Loader=yaml.FullLoader)
    # if the model is directly given in the yaml, then overwrite the model_conf
    if 'model' in config:
        model_conf = config['model']
    # if the config is still None, then raise an error   
    if model_conf is None:
        raise ValueError("model configuration should be either given in the yaml or in the config_dir")
    
    # Instantiate the model
    # change the device of the model to device
    
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = dy.eval(model_conf['class_path'])(**model_conf['init_args']).to(device)
    # load the model weights from the checkpoint
    if config['checkpoint_dir'] is not None:
        model.load_state_dict(torch.load(config['checkpoint_dir'])['module_state_dict'])
        
    
    # set to evaluation mode to get rid of any randomness happening in the 
    # architecture such as dropout
    model.eval()
    return model