import numpy as np

def load_data(cfg, train=True):
    dataset = cfg["dataset"].lower()
    if dataset == "random-gaussian":
        rng = np.random.default_rng(cfg['seed'])
        dset = rng.standard_normal(cfg["shape"])

        num_samples = cfg["shape"][0]
        dimension = cfg["shape"][1]
        print(f"Generating {num_samples} datapoints of dimension {dimension} normally distributed")

    elif dataset == "random-uniform":
        rng = np.random.default_rng(cfg['seed'])
        dset = rng.uniform(size=cfg["shape"])

        num_samples = cfg["shape"][0]
        dimension = cfg["shape"][1]
        print(f"Generating {num_samples} datapoints of dimension {dimension} uniformly distributed")

    else:
        raise Exception("Dataset not understood")

    return dset
