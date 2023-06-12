from ..two_step import TwoStepComponent
import typing as th
import functools

class DensityEstimator(TwoStepComponent):

    def __init__(
        self, 
        flatten=False, 
        data_shape=None, 
        denoising_sigma=None, 
        dequantize=False, 
        scale_data=False, 
        whitening_transform=False, 
        logit_transform=False, 
        clamp_samples=False
    ):
        super().__init__(flatten, data_shape, denoising_sigma, dequantize, scale_data, whitening_transform, logit_transform, clamp_samples)
            
        self._mid_level_representations = []
        self._representation_modules = []
        self._rank_to_index = {}
        
    def add_representation_module(self, module, rank: int = 0):
        # keep a list of sorted modules according to rank in self._representation_modules
        # and register forward hooks on them
        
        if not hasattr(self, '_representation_modules'):
            self._representation_modules = []
        
        self._representation_modules.append((rank, module))
        self._representation_modules = sorted(self._representation_modules, key=lambda x: x[0])
        
        # find the index of the module in the sorted list
        index = 0
        for i in range(len(self._representation_modules)):
            if self._representation_modules[i][1] == module:
                index = i
                break
        self._rank_to_index[rank] = index
        
        # remove all the hooks from the modules
        if hasattr(self, 'handles'):
            for handle in self.handles:
                handle.remove()
        
        # define the following forward hook
        def forward_hook(module, args, output, rank):
            repr_output = self.representation_hook(module, args, output, rank)
            self._mid_level_representations.append(repr_output)
            return output
        self.handles = []
        # When registering the modules in order of their increasing
        # rank, we can be sure that the forward hooks will be called
        # in the same order as the modules are registered
        for module in self._representation_modules:
            self.handles.append(module[1].register_forward_hook(functools.partial(forward_hook, rank=rank)))
    
    def representation_hook(self, module, args, output, module_rank):
        """
        This is a function that gets called on each of the hooked modules
        When registering the hook for every single module, we pass it into
        this function to get the representation output of the module.
        
        Here, this function simply returns the output of the module, in default.
        However, this function can be overridden to return a different representation.
        """
        return output
    
    def clear_representation(self):
        """
        Clears the buffer of representations.
        """
        self._mid_level_representations.clear()
    
    def get_representation(self, rank: th.Optional[int] = None):
        """
        Either returns the set of all representations, or the representation
        at the given index.
        
        You can call this function after calling the forward pass of the model
        which adds all the representations to the buffer.
        
        Args:
            ind (th.Optional[int], optional): Defaults to None.

        Returns:
            The representation at the given index, or the set of all representations.
        """
        if len(self._mid_level_representations) == 0:
            raise RuntimeError("No representations found. Make sure you call the forward pass of the model before calling this function.")
        if not hasattr(self, '_saved_length'):
            self._saved_length = len(self._mid_level_representations)
        if len(self._mid_level_representations) != self._saved_length:
            raise RuntimeError("The number of representations has changed!\n"
                               "Make sure you call the clear_representations before calling this function.")
        if rank is not None:
            if rank not in self._rank_to_index:
                raise ValueError(f"Rank {rank} not found.")
            ind = self._rank_to_index[rank]
            return self._mid_level_representations[ind]
        return self._mid_level_representations
                 
    def sample(self, n_samples):
        raise NotImplementedError("sample not implemented")

    def log_prob(self, x, **kwargs):
        raise NotImplementedError("log_prob not implemented")

    def loss(self, x, **kwargs):
        return -self.log_prob(x, **kwargs).mean()