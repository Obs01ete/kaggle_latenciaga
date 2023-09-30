import numpy as np
import torch
from torchmetrics.functional.regression import kendall_rank_corrcoef


def test_diff_matrix():
    N = 10
    latencies = np.random.random((N,)) * 10
    latencies_bc = np.tile(latencies, (N, 1))
    diff_matrix = latencies_bc - latencies_bc.T

    gt_argsort = np.argsort(latencies)

    sum_rows = np.sum(diff_matrix, axis=0)

    pred_argsort = np.argsort(sum_rows)

    print("gt_argsort=", gt_argsort)
    print("pred_argsort=", pred_argsort)

    kendall = kendall_rank_corrcoef(torch.tensor(pred_argsort),
                                    torch.tensor(gt_argsort))

    assert isinstance(kendall, torch.Tensor)

    kendall = kendall.item()
    print("kendall=", kendall)

    assert kendall > 1 - 1e-6

    print("Done")


if __name__ == "__main__":
    test_diff_matrix()
