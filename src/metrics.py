import torch


def tile_topk_metric(preds: torch.Tensor, target: torch.Tensor, k: int = 5):
    """Compute TileTopK metric.

    :param preds: (num_configs, )
    :param target: (num_configs, )
    :param k:
    :return: metric [-inf, 1]
    """
    bs = preds.shape[0]
    if bs < k:
        # there are several graphs with little number of configs
        k = bs
    best_runtime = target.min(0).values
    pred_bottomk_indices = torch.topk(preds, k=k, largest=False).indices
    predicted_runtimes = target[pred_bottomk_indices]
    best_predicted_runtimee = predicted_runtimes.min(0).values
    return 2 - best_predicted_runtimee / best_runtime

