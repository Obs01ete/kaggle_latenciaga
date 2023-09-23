import os
import numpy as np
from pathlib import Path

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.data.dataset import Dataset


class LayoutData(Dataset):
    # Compiler optimization
    OPTIM = ["layout", "tile"]
    # Source
    SRC = ["xla", "nlp"]
    # Search strategy
    SEARCH = ["default", "random"]
    # Dataset split
    SPLIT = ["train", "valid", "test"]
    # Collection
    COLL = [
        "layout-nlp-default",
        "layout-nlp-random",
        "layout-xla-default",
        "layout-xla-random",
        "tile-xla"
    ]

    def __init__(
            self,
            data_root: Path,
            coll: str,
            split: str,
            microbatch_size: int,
            ):

        super().__init__()

        assert os.path.exists(data_root)

        self.data_root = data_root
        assert coll in self.COLL
        self.coll = coll
        assert split in self.SPLIT
        self.split = split

        self.microbatch_size = microbatch_size

        self.data_dir = self._get_coll_root(coll)
        file_name_list = []
        for file in os.listdir(self.data_dir):
            if not file.endswith(".npz"): continue
            data_file = str(self.data_dir/file)
            file_name_list.append(data_file)
        self.file_name_list = file_name_list
    
    def len(self) -> int:
        return len(self.file_name_list)

    def _get_single(self, idx: int) -> Data:
        npz_dict = dict(np.load(self.file_name_list[idx]))
        data = Data(edge_index=torch.tensor(npz_dict["edge_index"].T).long())
        for name, array in npz_dict.items():
            if name == "edge_index":
                continue
            tensor = torch.tensor(array) # inherit type
            setattr(data, name, tensor)
        return data
    
    def get(self, idx: int) -> Batch:
        single_data = self._get_single(idx)

        # 'Data(node_feat=[372, 140], node_opcode=[372], edge_index=[2, 597], 
        # node_config_feat=[47712, 26, 18], node_config_ids=[26],
        # config_runtime=[47712], node_splits=[1, 3])'

        RUNTIME_SCALE_TO_SEC = 1e-9

        num_configs = single_data.node_config_feat.shape[0]
        chosen = np.random.choice(np.arange(num_configs),
                                  self.microbatch_size,
                                  replace=False)
        data_list = []
        for imb in range(self.microbatch_size):
            chosen_node_config_feat = single_data.node_config_feat[chosen]
            chosen_config_runtime = single_data.config_runtime[chosen]

            data = Data(edge_index=single_data.edge_index)
            data.node_feat = single_data.node_feat
            data.node_opcode = single_data.node_opcode
            data.node_config_feat = chosen_node_config_feat[imb]
            data.node_config_ids = single_data.node_config_ids
            data.config_runtime = RUNTIME_SCALE_TO_SEC * chosen_config_runtime[imb]
            # node_splits not going to use

            data_list.append(data)

        microbatch = Batch.from_data_list(data_list)

        # ignore type warning that Data must be returned.
        # it is passed through out of __getitem__.
        return microbatch

    def _get_coll_root(self, coll: str) -> Path:
        """Parse the collection and return the corresponding data root.
        
        Parameters:
            coll: collection

        Return
            data_root: data root of the collection
        """
        coll_terms = coll.split("-")
        if len(coll_terms) == 3:
            optim, src, search = coll_terms
            assert search in self.SEARCH
            data_root = self.data_root/f"{optim}/{src}/{search}/{self.split}"
        else:
            optim, src = coll_terms
            data_root = self.data_root/f"{optim}/{src}/{self.split}"
        
        assert optim in self.OPTIM
        assert src in self.SRC

        return data_root
