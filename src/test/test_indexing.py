import numpy as np
import torch


def test_cfg_feat_layout():
    num_nodes = 10
    config_feat_size = 2
    node_config_ids = torch.tensor([1, 3, 5, 7, 9], dtype=torch.int64)
    source_node_config_feat = torch.tensor([[1, 2],
                                            [3, 4],
                                            [5, 6],
                                            [7, 8],
                                            [9, 10]], dtype=torch.float32)
    final_node_config_feat = torch.zeros(size=(num_nodes, config_feat_size), dtype=torch.float)
    final_node_config_feat[node_config_ids, :] = source_node_config_feat
    gt_array = np.array([[0, 0],
                         [1, 2],
                         [0, 0],
                         [3, 4],
                         [0, 0],
                         [5, 6],
                         [0, 0],
                         [7, 8],
                         [0, 0],
                         [9, 10]],
                        dtype=float)
    np.testing.assert_array_equal(final_node_config_feat.cpu().detach().numpy(), gt_array)


if __name__ == "__main__":
    test_cfg_feat_layout()
