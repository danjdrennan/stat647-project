import torch
import numpy

__all__ = ["DataProcessor", "reverse_order"]


def reverse_order(order):
        """
        Reverses a maximin ordering.
        """
        return numpy.argsort(order)

class DataProcessor:
    """
    Preprocessor class to use with Vecchia approximation type models.
    Lets the user specify a sorting and neighbor search strategy, then
    provides functions to preprocess data using those strategies.
    """
    def __init__(self, sort_strategy, neighbor_strategy):
        self.sort_data = sort_strategy
        self.get_neighbors = neighbor_strategy

    @staticmethod
    def _is_array(array):
        dtype = type(array)
        if not isinstance(array, (numpy.ndarray, torch.Tensor)):
            raise ValueError(f"Array must be a numpy array or torch tensor, not {dtype}")
    
    @staticmethod
    def center(array):
        """
        Centers a multidimensional array on its last axis.
        """
        DataProcessor._is_array(array)
        return array - array.reshape(-1, array.shape[-1]).mean(0)

    @staticmethod
    def scale(array):
        """
        Scales a multidimensional array on its last axis.
        """
        DataProcessor._is_array(array)
        return array / array.reshape(-1, array.shape[-1]).std(0)
    
    def preprocess_data(self, locs, data, to_torch = True, center = False, scale = False):
        """
        Preprocessor for a training dataset.
        - Uses a maximin ordering (Guiness, 2018) to sort locations
        - Finds nearest neighbors in a Euclidean sense
        - Optionally centers and scales the data

        Returns a tuple of (ordered_locs, ordered_data, neighbors) 
        """
        order = self.sort_data(locs)
        ordered_locs = locs[order]
        ordered_data = data[:, order]
        neighbors = self.get_neighbors(locs[order], m = data.shape[1])
        if center:
            data = self.center(ordered_data)
        if scale:
            data = self.scale(ordered_data)
        
        if to_torch:
            return (
            torch.from_numpy(ordered_locs),
            torch.from_numpy(ordered_data), 
            torch.from_numpy(neighbors)
        )
        else:
            return (
                numpy.asarray(ordered_locs),
                numpy.asarray(ordered_data), 
                numpy.asarray(neighbors)
            )