import copy
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, Union, Any, Sequence, List
from enum import Enum
from functools import lru_cache
from tqdm import tqdm
from numpy.lib.npyio import NpzFile
import multiprocessing
from functools import partial

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.data.dataset import Dataset
from torch_geometric.utils import to_undirected

from src.utils import make_diff_matrix, triu_vector
from src.sys_utils import process_init_fn
from src.entropy_filter import EntropyFilter


def random_sample(d: Dict[Any, Any], num) -> Dict[Any, Any]:
    indices = list(np.random.choice(np.arange(len(d)), size=num))
    indices.sort()
    do = {i: d[idx] for i, idx in enumerate(indices)}
    return do


def random_sample_indices(config_indices: np.ndarray,
                          max_configs: int) -> np.ndarray:
    if len(config_indices) <= max_configs:
        return config_indices
    ii = np.random.choice(np.arange(len(config_indices)), max_configs)
    ii.sort()
    return_indices = config_indices[ii]
    return return_indices


def delete_dupl(npz_dict: dict, is_tile: bool) -> dict:
    npz_dict_copy = {}
    for k in npz_dict:
        npz_dict_copy[k] = npz_dict[k]
    if is_tile:
        src_shape = list(npz_dict_copy["config_feat"].shape)
        src_df = pd.DataFrame(npz_dict_copy["config_feat"])
        src_df["config_runtime"] = npz_dict_copy["config_runtime"]
        src_df["config_runtime_normalizers"] = npz_dict_copy["config_runtime_normalizers"]
    else:
        src_shape = list(npz_dict_copy["node_config_feat"].shape)
        reshaped_feats = npz_dict_copy["node_config_feat"].reshape(src_shape[0], src_shape[1] * src_shape[2])
        src_df = pd.DataFrame(reshaped_feats)
        src_df["config_runtime"] = npz_dict_copy["config_runtime"]
        src_df["config_runtime_normalizers"] = 1

    columns_for_filter = src_df.columns[:-2]
    src_df["ratio"] = src_df["config_runtime"] / src_df["config_runtime_normalizers"]
    src_df.sort_values(by=["ratio", "config_runtime"], inplace=True)
    src_df.drop_duplicates(subset=columns_for_filter, keep="first", inplace=True)
    src_shape[0] = src_df.shape[0]

    npz_dict_copy["config_runtime"] = src_df["config_runtime"].values
    if is_tile:
        npz_dict_copy["config_feat"] = src_df[columns_for_filter].values.reshape(src_shape)
        npz_dict_copy["config_runtime_normalizers"] = src_df["config_runtime_normalizers"].values
    else:
        npz_dict_copy["node_config_feat"] = src_df[columns_for_filter].values.reshape(src_shape)
    return npz_dict_copy


def repack_one_npz(src_file: str,
                   dst_file: str,
                   is_tile: bool,
                   is_random: bool,
                   filter_by_entropy: bool = False,
                   delete_duplicates: bool = False):

    npz = np.load(src_file)
    npz_dict = dict(npz)

    if filter_by_entropy and not is_tile:
        if is_random: # both xla and nlp
            entropy_thr = 0.3
        else: # is default, both xla and nlp
            entropy_thr = 3.0
        entropy_filter = EntropyFilter(npz_dict, entropy_thr=entropy_thr)
        if entropy_filter.picked_frac < 0.1:
            print(f"Most (90%+) of the graph data is corrupted, skip it: {src_file}")
            return
        maybe_npz_dict = entropy_filter.apply_filter()
        if maybe_npz_dict is None:
            print(f"No high-entropy sequences of blocks found, skip it: {src_file}")
            return
        print(f"Repacking with {entropy_filter.picked_frac:.3f} preserved: {src_file}")
        npz_dict = maybe_npz_dict

    if delete_duplicates:
        npz_dict = delete_dupl(npz_dict, is_tile)
        if npz_dict["config_runtime"].shape[0] == 1:
            # all configs are duplicates
            return

    if is_tile:
        np.savez(dst_file, **npz_dict)
    else:
        new_data = {k: v for k, v in npz_dict.items() if k != 'node_config_feat'}
        node_config_feat = npz_dict['node_config_feat']
        new_data['num_configs'] = np.array([node_config_feat.shape[0]], dtype=int)
        for i in range(node_config_feat.shape[0]):
            node_config_feat_ith = np.ascontiguousarray(node_config_feat[i])
            new_data[f"node_config_feat_{i}"] = node_config_feat_ith
        np.savez_compressed(dst_file, **new_data, allow_pickle=False)


class Purpose(Enum):
    Train = 1
    Valid = 2
    Test = 3


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

    FOLLOW_BATCH_KEYS = ["node_config_feat",
                         "node_config_ids",
                         "config_feat"]

    def __init__(
            self,
            data_root: Path,
            coll: str,
            split: str,
            purpose: Purpose,
            microbatch_size: Optional[int] = None,
            oversample_factor: Optional[int] = None,
            convert_to_undirected: bool = False,
            repack_npz: bool = True,
            cache_root: Path = Path("data_cache"),
            num_repack_processes: int = 16,
            delete_duplicates: bool = False,
            filter_by_entropy: bool = False,
            ):

        super().__init__()

        assert os.path.exists(data_root)

        if purpose in {Purpose.Train}:
            assert microbatch_size is not None
            # assert oversample_factor is not None
        if split in {Purpose.Valid, Purpose.Test}:
            assert microbatch_size is None
            assert oversample_factor is None

        assert coll in self.COLL
        self.coll = coll
        assert split in self.SPLIT + ["trainval"]
        self.split = split
        self.purpose = purpose

        self.num_repack_processes = num_repack_processes

        is_layout_xla = "layout-xla-" in self.coll

        self.is_tile = "tile-" in self.coll
        self.is_random = "-random" in self.coll
        self._repack_one_npz_partial = partial(repack_one_npz,
                                               is_tile=self.is_tile,
                                               is_random=self.is_random,
                                               delete_duplicates=delete_duplicates,
                                               filter_by_entropy=filter_by_entropy)

        MAX_CONFIGS_PER_GRAPH = 1000 # both layout and tile

        self.microbatch_size = microbatch_size

        self.convert_to_undirected = convert_to_undirected

        orig_data_dirs: List[Path]
        if self.split == "trainval":
            orig_data_dirs = [data_root / self._get_coll_subpath(curr_split)
                                     for curr_split in ("train", "valid")]
            for dir in orig_data_dirs:
                assert os.path.exists(dir), f"Cannot find {dir}"
        else:
            orig_data_dirs = [data_root / self._get_coll_subpath(self.split)]
            assert os.path.exists(orig_data_dirs[0]), f"Cannot find {orig_data_dirs}"

        self.data_dir: Path
        if repack_npz:
            self.data_dir = cache_root / self._get_coll_subpath(self.split)
            print(f"Going with repacked npz's from {self.data_dir}")
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir, exist_ok=True)
                self._repack_data(orig_data_dirs, self.data_dir)
        else:
            print("Going with original (non-repacked) npz's")
            assert len(orig_data_dirs) > 1, \
                f"Cannot use this spllit {self.split} without repacking {orig_data_dirs}"
            self.data_dir = orig_data_dirs[0]

        file_name_list = []
        for file in os.listdir(self.data_dir):
            if not file.endswith(".npz"):
                continue
            data_file = str(self.data_dir/file)
            file_name_list.append(data_file)

        if is_layout_xla:
            if split in {"train", "trainval"}:
                file_name_list = [v for v in file_name_list if not "unet" in v]
            else:
                pass

        if oversample_factor is not None:
            self.file_name_list = file_name_list * oversample_factor
        else:
            self.file_name_list = file_name_list

        self._len: int
        self.map_idx_to_name_and_config: \
            Optional[Dict[int, Dict[str, Any]]]
        if self.purpose == Purpose.Train:
            self.map_idx_to_name_and_config = None
            self._len = len(self.file_name_list)
        else:
            idx = 0
            self.map_idx_to_name_and_config = dict()
            for file_path in tqdm(self.file_name_list):
                npz = np.load(file_path)
                num_configs = npz["config_runtime"].shape[0]
                config_indices = np.arange(num_configs)
                if purpose != Purpose.Test: # ie Purpose.Valid
                    config_indices = random_sample_indices(config_indices,
                                                           MAX_CONFIGS_PER_GRAPH)
                for config_idx in config_indices:
                    self.map_idx_to_name_and_config[idx] = dict(
                        file_path=file_path,
                        config_idx=config_idx)
                    idx += 1

            self._len = len(self.map_idx_to_name_and_config)
        pass

    def _repack_data(self, orig_data_dirs: Sequence[Path], cache_dir: Path):
        src_dst_list = []
        for orig_data_dir in orig_data_dirs:
            for file in tqdm(os.listdir(orig_data_dir)):
                if not file.endswith(".npz"):
                    continue
                src_file = str(orig_data_dir/file)
                dst_file = str(cache_dir/file)
                src_dst_list.append((src_file, dst_file))

        print("Repacking npz's...")
        enable_multiprocessing = True
        if enable_multiprocessing:
            with multiprocessing.Pool(self.num_repack_processes, initializer=process_init_fn) as pool:
                pool.starmap(self._repack_one_npz_partial, src_dst_list)
        else:
            for src_file, dst_file in src_dst_list:
                self._repack_one_npz_partial(src_file, dst_file)
        print("Done repacking npz's.")

    def len(self) -> int:
        return self._len

    @lru_cache(maxsize=1)
    def _get_single(self, file_path: str) -> Tuple[Data, NpzFile]:
        npz = np.load(file_path)

        edge_index = torch.tensor(npz["edge_index"].T).long()

        if self.convert_to_undirected:
            edge_index, edge_labels = self._convert_graph_to_undirected(edge_index)
            data = Data(edge_index=edge_index, edge_labels=edge_labels)
        else:
            data = Data(edge_index=edge_index)

        for name in npz.keys():
            if name == "edge_index":
                continue
            if name == 'num_configs':
                continue
            if "node_config_feat_" in name:
                continue
            array = npz[name]
            tensor = torch.tensor(array) # inherit type
            setattr(data, name, tensor)

        fname_wo_ext = os.path.splitext(os.path.basename(file_path))[0]
        data.fname = fname_wo_ext
        return data, npz
    
    def get(self, idx: int) -> Batch: # not Data

        RUNTIME_SCALE_TO_SEC = 1e-9

        if self.map_idx_to_name_and_config is None: # train
            assert self.microbatch_size is not None

            microbatch_size = self.microbatch_size

            file_path = self.file_name_list[idx]
            single_data, npz = self._get_single(file_path)

            # 'Data(node_feat=[372, 140], node_opcode=[372], edge_index=[2, 597], 
            # node_config_feat=[47712, 26, 18], node_config_ids=[26],
            # config_runtime=[47712], node_splits=[1, 3])'

            num_configs: int
            if self.is_tile:
                num_configs = single_data.config_feat.shape[0]
            else:
                if 'node_config_feat' in single_data.keys:
                    num_configs = single_data.node_config_feat.shape[0]
                else:
                    num_configs = npz['num_configs'].item()

            # enabled replace=True for all, so that tile does not crash
            chosen = np.random.choice(np.arange(num_configs),
                                      microbatch_size,
                                      replace=True)
            if self.is_tile:
                chosen_config_feat = single_data.config_feat[chosen]
            else:
                if 'node_config_feat' in single_data.keys:
                    chosen_config_feat = single_data.node_config_feat[chosen]
                else:
                    chosen_config_feat_np = np.stack([npz[f"node_config_feat_{idx}"]
                                                      for idx in chosen])
                    chosen_config_feat = torch.tensor(chosen_config_feat_np)
            chosen_config_runtime = single_data.config_runtime[chosen]
            chosen_config_runtime_sec = RUNTIME_SCALE_TO_SEC * chosen_config_runtime

            if self.is_tile:
                normalized_runtime = (single_data.config_runtime[chosen] /
                                      single_data.config_runtime_normalizers[chosen])
                diff_matrix = make_diff_matrix(normalized_runtime).unsqueeze(0)
                diff_triu_vector = triu_vector(diff_matrix.squeeze(0)).unsqueeze(0)
            else:
                diff_matrix = make_diff_matrix(chosen_config_runtime_sec).unsqueeze(0)
                diff_triu_vector = triu_vector(diff_matrix.squeeze(0)).unsqueeze(0)

            data_list = []
            for imb in range(self.microbatch_size):
                data = Data(edge_index=single_data.edge_index)
                data.node_feat = single_data.node_feat
                data.node_opcode = single_data.node_opcode
                if self.is_tile:
                    data.config_feat = chosen_config_feat[imb]
                    data.config_runtime = single_data.config_runtime[chosen][imb] \
                        / single_data.config_runtime_normalizers[chosen][imb]
                else:
                    data.node_config_feat = chosen_config_feat[imb]
                    data.node_config_ids = single_data.node_config_ids
                    data.config_runtime = chosen_config_runtime_sec[imb]
                # We have to put the diff_matrix in every sample
                # of the microbatch for batching to work correctly.
                data.diff_triu_vector = diff_triu_vector
                data.fname = single_data.fname
                if self.convert_to_undirected:
                    data.edge_labels = single_data.edge_labels
                # node_splits not going to use

                data_list.append(data)

            microbatch = Batch.from_data_list(data_list, follow_batch=self.FOLLOW_BATCH_KEYS)
        else: # val and test
            name_and_config = self.map_idx_to_name_and_config[idx]
            file_path = name_and_config['file_path']
            # print(self._get_single.cache_info())
            # print("getting", file_path)
            single_data, npz = self._get_single(file_path)
            config_idx = name_and_config['config_idx']

            data = Data(edge_index=single_data.edge_index)
            data.node_feat = single_data.node_feat
            data.node_opcode = single_data.node_opcode

            if self.is_tile:
                data.config_feat = single_data.config_feat[config_idx]
                data.config_runtime = (single_data.config_runtime[config_idx] /
                                       single_data.config_runtime_normalizers[config_idx])
            else:
                if 'node_config_feat' in single_data.keys:
                    data.node_config_feat = single_data.node_config_feat[config_idx]
                else:
                    node_config_feat_np = npz[f"node_config_feat_{config_idx}"]
                    data.node_config_feat = torch.tensor(node_config_feat_np)
                data.node_config_ids = single_data.node_config_ids
                data.config_runtime = RUNTIME_SCALE_TO_SEC * single_data.config_runtime[config_idx]
            data.fname = single_data.fname
            if self.convert_to_undirected:
                data.edge_labels = single_data.edge_labels

            microbatch = Batch.from_data_list([data], follow_batch=self.FOLLOW_BATCH_KEYS)

        # ignore type warning that Data must be returned.
        # it is passed through out of __getitem__.
        return microbatch # type: ignore

    def _get_coll_subpath(self, split: str) -> Path:
        """Parse the collection and return the corresponding data root.
        
        Parameters:
            split: split

        Return
            data_root: data root of the collection
        """
        coll_terms = self.coll.split("-")
        if len(coll_terms) == 3:
            optim, src, search = coll_terms
            assert search in self.SEARCH
            subpath = Path(f"{optim}/{src}/{search}/{split}")
        else:
            optim, src = coll_terms
            subpath = Path(f"{optim}/{src}/{split}")
        
        assert optim in self.OPTIM
        assert src in self.SRC

        return subpath

    @staticmethod
    def _convert_graph_to_undirected(directed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert to undirected
        undirected = to_undirected(directed)
        assert isinstance(undirected, torch.Tensor)

        # Create edge labels for the undirected graph
        edge_labels = torch.ones(undirected.size(1), dtype=torch.float32)

        # For every [v, u] edge in the undirected graph, set the label to -1
        edge_map = set((u.item(), v.item()) for u, v in zip(directed[0], directed[1]))
        for i, (u, v) in enumerate(zip(undirected[0], undirected[1])):
            if (v.item(), u.item()) in edge_map:
                edge_labels[i] = -1

        return undirected, edge_labels
