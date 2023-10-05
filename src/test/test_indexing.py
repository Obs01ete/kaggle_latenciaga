import numpy as np
import torch

from torch.utils.data import DataLoader

from src.data import LayoutData

from pathlib import Path


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

def test_dataloader_to_undirected():
    data_root = Path("/home/khizbud/latenciaga/data/npz_all/npz")

    collection = "layout-xla-random"

    val_data = LayoutData(
        data_root,
        coll=collection,
        split="valid",
        convert_to_undirected=True)

    # Load updated data and check dimensions
    indirect_data = val_data[0]
    assert indirect_data.edge_index.size(1) == indirect_data.edge_labels.size(0)

    # Load manually and check that data is augmented
    direct_data = dict(np.load(val_data.file_name_list[0]))
    assert "edge_labels" not in direct_data
    assert indirect_data.edge_index.size(1) == direct_data["edge_index"].shape[0] * 2

def test_conversion_to_undirected_1():
    directed = torch.tensor([[0., 2., 4.],
                             [1., 3., 5.]], dtype=torch.float32)

    undirected, directions = LayoutData._convert_graph_to_undirected(directed)
    
    assert True == torch.equal(undirected, 
                               torch.tensor([[0., 1., 2., 3., 4., 5.],
                                             [1., 0., 3., 2., 5., 4.]], dtype=torch.float32))

    assert True == torch.equal(directions, 
                               torch.tensor([ 1., -1.,  1., -1.,  1., -1.], dtype=torch.float32))

def test_conversion_to_undirected_2():
    directed = torch.tensor([[0., 0., 0., 0., 2.],
                             [1., 2., 3., 4., 3.]], dtype=torch.float32)

    undirected, directions = LayoutData._convert_graph_to_undirected(directed)
    
    assert True == torch.equal(undirected, 
                               torch.tensor([[0., 0., 0., 0., 1., 2., 2., 3., 3., 4.],
                                             [1., 2., 3., 4., 0., 0., 3., 0., 2., 0.]], dtype=torch.float32))

    assert True == torch.equal(directions, 
                               torch.tensor([ 1.,  1.,  1.,  1., -1., -1.,  1., -1., -1., -1.], dtype=torch.float32))


if __name__ == "__main__":
    test_cfg_feat_layout()
    test_dataloader_to_undirected()
    test_conversion_to_undirected_1()
    test_conversion_to_undirected_2()
