import torch
from torch.nn import Parameter, ParameterList

import gpytorch
from gpytorch.models import GP
from pyro.distributions import InverseGamma, Normal

from ..utils import DataProcessor, reverse_order as _reverse_order

__all__ = ["GraphicalNostationaryGP"]

__doc__ = """
BNNCE: Bayesian Nonstationary and Nonparametric Covariance Estimation is an
implementation of the work in Kidd and Katzfuss (2022, Bayesian Analysis)
using Gpytorch.
"""

def _make_design_matrix(Y: torch.TensorType, m: int, neighbors: torch.TensorType) -> torch.TensorType:
    """
    Finds design matrix of nearest neighbors for a given location i in y.shape[1].
    Batches over columns of Y, yielding a tensor of shape (N, m, n) (Y is n x N).
    """
    # Use cm to truncate neighbors
    truncated_neighbors = neighbors[:, :m]
    return Y.T[truncated_neighbors].swapdims(1, -1)

def _get_m(theta: ParameterList, vec_size = 900, threshold = 0.001):
    """
    Finds the largest integer n in {1, ..., vec_size} such that exp(-theta * n) >= threshold.
    """
    return torch.arange(vec_size).add(1).mul(-theta[2]).exp().ge(threshold).sum().item()

def _scale(theta: ParameterList, data: torch.TensorType, dim: int = 2) -> torch.TensorType:
    """Scales parameter by location."""
    index = torch.arange(data.shape[1]).add(1)
    return theta[0] * (1 - index.pow(-1 / dim).mul(-theta[1]).exp())

def _make_V(theta: ParameterList, data: torch.TensorType, neighbors: torch.TensorType, m: int, **kwargs) -> torch.TensorType:
    """Makes prior covariance at location loc (i in paper)."""
    n, N = data.shape
    V = torch.arange(m).add(1).mul(-theta[2]).exp().diag().repeat(N, 1, 1)
    return V

def _set_alpha(theta: ParameterList, locs: torch.TensorType, **kwargs) -> torch.TensorType:
    """Sets prior alpha at all locations."""
    return torch.ones_like(locs).mul(6)

def _set_beta(theta: ParameterList, locs: torch.TensorType, **kwargs) -> torch.TensorType:
    """Sets prior beta at all locations."""
    return 5. * _scale(theta, locs, **kwargs)

class GraphicalNostationaryGP(GP, DataProcessor):
    """
    Provides a graphical model for a nonstationary GP to semiparametrically
    estimate the covariance function. The estimation generalizes a Vecchia
    approximation using a modified Cholesky decomposition.

    Class Instantation:
        - Data: (locs, data)
        - Sorting strategy
        - Neighbor search strategy
    """
    def __init__(self, locs, data, sort_strategy, neighbor_strategy):
        GP.__init__(self)
        DataProcessor.__init__(self, sort_strategy, neighbor_strategy)
        self._locs = locs
        self._data = data
        self.theta = ParameterList()
        self.theta.append(Parameter(torch.tensor(data.mean())))
        self.theta.append(Parameter(torch.tensor(data.std()).log()))
        self.theta.append(Parameter(torch.tensor(0.2305)))

    def _transform_data(self, **kwargs) -> None:
        """Transforms data to be used in the GP."""
        t = self.preprocess_data(self._locs, self._data, to_torch = True, **kwargs)
        self.locs = t[0].float()
        self.data = t[1].float()
        self.neighbors = t[2][:, 1:].long()

    def forward(self, y, **kwargs):
        """
        Computes and returns
        - prior alpha
        - posterior alpha
        - prior beta
        - posterior beta
        - prior V
        - posterior G

        Use the outputs to minimize the negative log likelihood.
        """
        n, N = y.shape
        m = _get_m(self.theta, N, threshold = 0.001)
        X = _make_design_matrix(self.data, m, self.neighbors)
        scale = _scale(self.theta, self.data, **kwargs)
        V = _make_V(self.theta, self.data, self.neighbors, m, **kwargs)
        G = torch.zeros_like(V)
        
        prior_alpha = _set_alpha(self.theta, self.data, **kwargs)
        prior_beta = _set_beta(self.theta, self.data, **kwargs)
        post_alpha = prior_alpha + n / 2
        post_beta = prior_beta

        for i in range(N):
            G[i] = torch.inverse(X[i].T @ X[i] + V[i] / scale[i])
            inv = torch.inverse(torch.eye(n) + X[i] @ V[i] @ X[i].T / scale[i])
            post_beta[i] += y[:, i].T @ inv @ y[:, i] / 2
        
        return prior_alpha, post_alpha, prior_beta, post_beta, V, G

    def likelihood(self, prior_alpha, post_alpha, prior_beta, post_beta, V, G):
        """
        Compute log score of the GP for model training.
        """
        return torch.sum(
            0.5 * (torch.logdet(G) - torch.logdet(V)) + \
            prior_alpha * torch.log(prior_beta) - post_alpha * torch.log(post_beta) + \
            torch.lgamma(post_alpha) - torch.lgamma(prior_alpha)
        )

    def fit(self, optimizer, num_epochs = 100, track_loss = True, **kwargs):
        self._transform_data(**kwargs)
        self.train()
        
        running_loss = []
        for i in range(num_epochs):
            optimizer.zero_grad()
            out = self(self.data)
            loss = -self.likelihood(*out)
            if track_loss:
                running_loss.append(loss.item())
            loss.backward()
            print(f'Epoch {i}: {loss.item():.4f}')
            optimizer.step()
            
        if track_loss:
            return running_loss
        
    def sample(self, n_samples):
        """
        Returns n_samples from the nonstationary GP in the ordering used.
        A reverse ordering strategy should be used to return the samples
        to the original ordering.
        """
        self.eval()
        with torch.no_grad():
            n, N = self.data.shape
            
            m = _get_m(self.theta[2], N, threshold = 0.001)
            X = _make_design_matrix(self.data, m, self.neighbors)
            components = self(self.data)
            G = components[-1]
            T = torch.zeros(N, N)
            D = torch.zeros(N)
            for i in range(N):
                Xi = next(X)
                t = G[i] @ Xi.T @ self.data[:, i]
                L = torch.cholesky(G[i], upper = False)
                d = InverseGamma(components[1][i], components[3][i]).sample()
                T[:, i] = L @ Normal(t, d.sqrt()).sample((n, ))
                D[i] = d
            # Now draw new samples using y* = (T^{-1})^{transpose}D^{1/2}z
            U = torch.inverse(T).T
            half_D = D.sqrt()
            z = Normal(0, 1).sample((N, n_samples))
        ystar = U @ (half_D * z)
        return ystar
                