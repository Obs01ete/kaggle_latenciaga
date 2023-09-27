import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, Union, Any
from functools import lru_cache
import random

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.data.dataset import Dataset


def random_sample(d: Dict[Any, Any], num) -> Dict[Any, Any]:
    indices = list(np.random.choice(np.arange(len(d)), size=num))
    indices.sort()
    do = {i: d[idx] for i, idx in enumerate(indices)}
    return do


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

    FOLLOW_BATCH_KEYS = ["node_config_feat", "node_config_ids"]

    MAX_VAL_SAMPLES = 200_000

    def __init__(
            self,
            data_root: Path,
            coll: str,
            split: str,
            microbatch_size: Optional[int] = None,
            oversample_factor: Optional[int] = None,
            ):

        super().__init__()

        assert os.path.exists(data_root)

        if split == "train":
            assert microbatch_size is not None
            # assert oversample_factor is not None
        if split in {"valid", "test"}:
            assert microbatch_size is None
            assert oversample_factor is None

        self.data_root = data_root
        assert coll in self.COLL
        self.coll = coll
        assert split in self.SPLIT
        self.split = split

        self.microbatch_size = microbatch_size

        self.data_dir = self._get_coll_root(coll)
        file_name_list = []
        for file in os.listdir(self.data_dir):
            if not file.endswith(".npz"):
                continue
            data_file = str(self.data_dir/file)
            file_name_list.append(data_file)

        if oversample_factor is not None:
            self.file_name_list = file_name_list * oversample_factor
        else:
            self.file_name_list = file_name_list

        self._len: int
        self.map_idx_to_name_and_config: \
            Optional[Dict[int, Dict[str, Union[int, str]]]]
        if self.split == "train":
            self.map_idx_to_name_and_config = None
            self._len = len(self.file_name_list)
        else:
            idx = 0
            self.map_idx_to_name_and_config = dict()
            for file_path in self.file_name_list:
                npz_dict = dict(np.load(file_path))
                num_configs = npz_dict["config_runtime"].shape[0]
                for config_idx in range(num_configs):
                    self.map_idx_to_name_and_config[idx] = dict(
                        file_path=file_path,
                        config_idx=config_idx)
                    idx += 1

            if len(self.map_idx_to_name_and_config) > self.MAX_VAL_SAMPLES:
                print(f"Random sampling val {len(self.map_idx_to_name_and_config)} "
                      f"to {self.MAX_VAL_SAMPLES} samples")
                self.map_idx_to_name_and_config = \
                    random_sample(self.map_idx_to_name_and_config,
                                  self.MAX_VAL_SAMPLES)
            else:
                print(f"Running on all validation samples")

            self._len = len(self.map_idx_to_name_and_config)
        pass

    def len(self) -> int:
        return self._len

    @lru_cache(maxsize=1)
    def _get_single(self, file_path: str) -> Data:
        npz_dict = dict(np.load(file_path))

        data = Data(edge_index=torch.tensor(npz_dict["edge_index"].T).long())

        for name, array in npz_dict.items():
            if name == "edge_index":
                continue
            tensor = torch.tensor(array) # inherit type
            setattr(data, name, tensor)

        fname_wo_ext = os.path.splitext(os.path.basename(file_path))[0]
        data.fname = fname_wo_ext
        return data
    
    def get(self, idx: int) -> Batch: # not Data

        RUNTIME_SCALE_TO_SEC = 1e-9

        if self.map_idx_to_name_and_config is None: # train
            assert self.microbatch_size is not None

            file_path = self.file_name_list[idx]
            single_data = self._get_single(file_path)

            # 'Data(node_feat=[372, 140], node_opcode=[372], edge_index=[2, 597], 
            # node_config_feat=[47712, 26, 18], node_config_ids=[26],
            # config_runtime=[47712], node_splits=[1, 3])'

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
                data.fname = single_data.fname
                # node_splits not going to use

                data_list.append(data)

            microbatch = Batch.from_data_list(data_list, follow_batch=self.FOLLOW_BATCH_KEYS)
        else: # val and test
            name_and_config = self.map_idx_to_name_and_config[idx]
            file_path = name_and_config['file_path']
            # print(self._get_single.cache_info())
            # print("getting", file_path)
            single_data = self._get_single(file_path)
            config_idx = name_and_config['config_idx']

            data = Data(edge_index=single_data.edge_index)
            data.node_feat = single_data.node_feat
            data.node_opcode = single_data.node_opcode
            data.node_config_feat = single_data.node_config_feat[config_idx]
            data.node_config_ids = single_data.node_config_ids
            data.config_runtime = RUNTIME_SCALE_TO_SEC * single_data.config_runtime[config_idx]
            data.fname = single_data.fname

            microbatch = Batch.from_data_list([data], follow_batch=self.FOLLOW_BATCH_KEYS)

        # ignore type warning that Data must be returned.
        # it is passed through out of __getitem__.
        return microbatch # type: ignore

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
