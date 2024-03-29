"""
utility functions being used for OOD detection models
"""
import torch

def buffer_loader(loader, buffer_size, limit=None):
    # tekes in a torch dataloader and returns an iterable where each
    # iteration returns a list of buffer_size batches
    for i, batch in enumerate(loader):
        if limit is not None and i // buffer_size >= limit:
            break
        if i % buffer_size == 0:
            if i != 0:
                yield buffer
            buffer = []
        buffer.append(batch)
    if len(buffer) > 0:
        yield buffer

def stack_back_iterables(reference_iterable, *chunky_iterables):
    """
    This function takes in an iterable of torch tensors and a set of other iterables of torch tensors.
    
    It is intended for the following use case:
    If reference_loader has a batch size of b and 
    all the current_loaders have broken down the batch size into smaller chunks of size [b_1, b_2, ..., b_n]
    then this function will return a set of iterables where all of the iterables have the same batch size as the reference_loader. 

    This happens for computation purposes, where you need to break up a batch even further; however, stacking them
    back is necessary for interpretation purposes.
    """
    stacked_back_iterables = [[] for _ in chunky_iterables]
    cumul_ref = 0
    cumul_current = [0 for _ in chunky_iterables]
    current_loaders = [iter(current_loader) for current_loader in chunky_iterables]
    
    for b in reference_iterable:
        cumul_ref += len(b)
        
        for i, current_loader in enumerate(current_loaders):
            intermediate = []
            while cumul_current[i] < cumul_ref:
                t = next(current_loader)
                
                
                intermediate.append(t)
                cumul_current[i] += len(t)
            if len(intermediate) > 0:
                intermediate = torch.cat(intermediate)
                stacked_back_iterables[i].append(intermediate)
            
    return tuple(stacked_back_iterables)
        