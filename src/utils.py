import numpy as np
import torch


def odd_pow_np(input: np.ndarray, exponent: float) -> np.ndarray:
    return np.sign(input) * np.power(np.abs(input), exponent)


# Does not work, produces NaNs because of infinite gradient around zero
def odd_pow(input: torch.Tensor, exponent: float) -> torch.Tensor:
    return input.sign() * input.abs().pow(exponent)


def log_compression(input: torch.Tensor) -> torch.Tensor:
    return input.sign() * input.abs().log1p()


def make_diff_matrix(latencies: torch.Tensor):
    assert len(latencies.shape) == 1
    latencies_bc = torch.tile(latencies.unsqueeze(0),
                              dims=(latencies.shape[0], 1))
    return latencies_bc - latencies_bc.transpose(1, 0)


def triu_vector(matrix: torch.Tensor) -> torch.Tensor:
    assert len(matrix.shape) == 2
    ind = torch.triu_indices(matrix.shape[0], matrix.shape[1], offset=1)
    triu_vec = matrix[ind[0], ind[1]]
    return triu_vec

