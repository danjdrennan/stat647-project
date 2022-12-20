import maxmin_cpp
import numpy
import torch


__all__ = ['maxmin_sort']

def maxmin_sort(locs_matrix: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of exact max min from GPvecchia:
    https://github.com/katzfuss-group/GPvecchia/blob/master/src/MaxMin.cpp.
    
    locs_matrix : (N x 2) torch tensor of locations. Does not support batches. 
    
    Returns (N,) torch tensor of location in max min ordering.
    """
    if isinstance(locs_matrix, numpy.ndarray):
        locs_matrix = torch.from_numpy(locs_matrix)
    if locs_matrix.dim() != 2 or locs_matrix.shape[1] != 2:
        raise ValueError("X must be (N x 2) tensor.")
    return maxmin_cpp.MaxMincpp(locs_matrix).type(torch.LongTensor) - 1
