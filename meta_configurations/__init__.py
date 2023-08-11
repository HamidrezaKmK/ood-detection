import typing as th
from pprint import pprint

def get_coupling(conf):
    """
    This function takes in a configuration and returns the coupling transform
    """
    if isinstance(conf, list):
        for c in conf:
            ret = get_coupling(c)
            if ret is not None:
                return ret
        return None
    
    if isinstance(conf, dict):
        for k, v in conf.items():
            if k == 'coupling_transform_cls':
                return v.split('.')[-1]
            else:
                ret = get_coupling(v)
                if ret is not None:
                    return ret
        return None
    
    return None

def change_coupling_layers(conf, coupling_name: str, additional_args: th.Optional[dict] = None):
    """
    This function takes in an entire configuration
    and replaces all the coupling transforms to the one specified.
    If the coupling transform is in need of additional arguments,
    this function will add them to the configuration.
    """
    if isinstance(conf, list):
        ret = []
        for c in conf:
            ret.append(change_coupling_layers(c, coupling_name=coupling_name, additional_args=additional_args))
        return ret
    
    if isinstance(conf, dict):
        ret = {}
        for k, v in conf.items():
            if k == 'coupling_transform_cls':
                ret = {}
                for k_, v_ in conf.items():
                    if k_ in ['net', 'mask']:
                        ret[k_] = v_
                ret['coupling_transform_cls'] = f'nflows.transforms.{coupling_name}'
                if additional_args:
                    ret.update(additional_args)
                return ret
            else:   
                ret[k] = change_coupling_layers(v, coupling_name=coupling_name, additional_args=additional_args)
        return ret
    
    return conf

def get_generated_config(conf: dict):
    return {
        'dataset': 'dgm-generated',
        'dgm_args': {
            'model_loading_config': conf['base_model'],
            'seed': 10,
            'length': 1000,
        }
    }

def run_name_changer(conf, run_name):
    ret = conf['data']['in_distribution']['dataloader_args']['dataset']
    ret += '_vs_'
    ret += conf['data']['out_of_distribution']['dataloader_args']['dataset']
    ret += f"_{conf['data']['out_of_distribution']['pick_loader']}"
    ret += f"_{run_name}"
    return ret
