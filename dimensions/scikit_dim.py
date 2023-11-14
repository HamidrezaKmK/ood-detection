import time
import skdim
from dimensions.base_method import BaseIntrinsicDimensionEstimationMethod
import typing as th
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

class SkdimIntrinsicDimensionEstimation(BaseIntrinsicDimensionEstimationMethod):
    def __init__(
        self,
        estimator_name: th.Literal[
            'corrint',
            'danco',
            'ess',
            'fishers',
            'knn',
            'lpca',
            'mada',
            'mind_ml',
            'mle',
            'mom',
            'tle',
            'twonn',
        ],
        *args,
        estimator_args: th.Optional[th.Dict[str, th.Any]] = None,
        verbose: int = 0,
        n_jobs: int = 0,
        **kwargs,
    ):
        # Call the parent class
        super().__init__(
            *args,
            **kwargs,
        )
        
        # Set the amount of verbosity
        self.verbose = verbose
        
        estimator_args = estimator_args or {}
        
        # for the estimator_name given, initialize
        self.estimator_name = estimator_name
        if estimator_name == 'corrint':
            self.estimator = skdim.id.CorrInt(**estimator_args)
        elif estimator_name == 'danco':
            self.estimator = skdim.id.DANCo(**estimator_args)
        elif estimator_name == 'ess':
            self.estimator = skdim.id.ESS(**estimator_args) #(ver=cfg['ver'], d=cfg['d'])
        elif estimator_name == 'fishers':
            self.estimator = skdim.id.FisherS(**estimator_args)
        elif estimator_name == 'knn':
            self.estimator = skdim.id.KNN(**estimator_args)        
        elif estimator_name == 'lpca':
            self.estimator = skdim.id.lPCA(**estimator_args)
        elif estimator_name == 'mada':
            self.estimator = skdim.id.MADA(**estimator_args)
        elif estimator_name == 'mind_ml':
            self.estimator = skdim.id.MiND_ML(**estimator_args)
        elif estimator_name == 'mle':
            self.estimator = skdim.id.MLE(**estimator_args)    
        elif estimator_name == 'mom':
            self.estimator = skdim.id.MOM(**estimator_args)
        elif estimator_name == 'tle':
            self.estimator = skdim.id.TLE(**estimator_args)
        elif estimator_name == 'twonn':
            self.estimator = skdim.id.TwoNN(**estimator_args)
        else:
            raise ValueError(f"Unknown estimator {estimator_name} provided.")

        self.n_jobs = n_jobs
    
    def fit(
        self,
        subsample_size: th.Optional[int] = None,
    ):
        if self.verbose > 0:
            print(f"Estimating dimension using {self.estimator_name}.")
        if self.test_dataloader is None:
            raise Exception("Cannot fit scikit learn module because no train dataloader is specified!")
        
        # Keep track of time
        start = time.time()
        
        # Iterate over the dataset to create an appropriate numpy array for fitting the scikit-learn model
        X = []
        curr = 0
        for x, y, idx in self.test_dataloader:
            curr += x.shape[0]
            X.append(x.cpu().numpy().reshape(x.shape[0], -1))
            if subsample_size is not None and curr >= subsample_size:
                break
        
        X = np.concatenate(X)
        
        # Fit the scikit-learn model
        
        self.estimator.fit(X)
        end = time.time()
        
        # Print the amount of time it took if the verbose argument is large
        if self.verbose > 0:
            print("Training took {:.2f} seconds.".format(end - start))
        
    def estimate_lid(self, x):
        # This is not really a LID estimate; however, 
        # these methods only work for general ID not LID values
        return np.array([self.estimator.dimension_ for _ in range(len(x))])
    
    def estimate_id(self):
        return self.estimator.dimension_