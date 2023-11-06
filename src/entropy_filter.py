from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np

# from scipy.stats import ttest_ind
# from scipy.special import kl_div
from scipy.stats import entropy


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


class EntropyFilter:
    def __init__(self,
                 npz_dict: Dict[str, np.ndarray],
                 entropy_thr: float,
                 block_size: int = 100,
                 smooth_window: int = 5,
                 config_slice = slice(None, None), # use only for plotting
                ):

        self.entropy_thr = entropy_thr
        self.block_size = block_size

        self.npz_dict = npz_dict
        runtime_raw = self.npz_dict["config_runtime"][config_slice]
        # node_config_feat = self.npz_dict["node_config_feat"][config_slice]
        # node_config_ids = self.npz_dict["node_config_ids"][config_slice]

        # It is important to normalize runtimes for `entropy` to work properly
        eps = 1e-3
        runtime = (runtime_raw - runtime_raw.min() + eps/2) / \
            (runtime_raw.max() - runtime_raw.min() + eps)
        # print(runtime.min(), runtime.max())

        num_blocks = len(runtime) // block_size
        prev_blk_runtime = None
        pvalue_kv: List[Tuple[int, float]] = []
        for block_idx in range(num_blocks):
            blk_runtime = runtime[block_idx*block_size:(block_idx+1)*block_size]
            if prev_blk_runtime is not None:
                pv = entropy(blk_runtime, prev_blk_runtime) # aka KL-divergence
                pvalue_kv.append((block_idx*block_size, pv.item()))
            prev_blk_runtime = blk_runtime

        # block_x = np.array([v[0] for v in pvalue_kv])
        block_y_raw = np.array([v[1] for v in pvalue_kv])
        # print("len(block_y_raw)", len(block_y_raw))
        block_y_raw[np.isinf(block_y_raw)] = float('nan')
        num_valid_blocks = np.count_nonzero(~np.isnan(block_y_raw))
        # print("num_valid_blocks", num_valid_blocks)
        
        self.block_entropy: Optional[np.ndarray]
        if num_valid_blocks > 0:
            self.block_entropy = smooth(block_y_raw, smooth_window)
        else:
            self.block_entropy = None

    @property
    def block_mask(self) -> Optional[np.ndarray]:
        return (self.block_entropy >= self.entropy_thr) \
            if self.block_entropy is not None else None

    @property
    def picked_frac(self) -> float:
        if self.block_mask is None:
            return 0.0
        return np.nansum(self.block_mask) / len(self.block_mask)

    def apply_filter(self) -> Optional[Dict[str, np.ndarray]]:
        if self.block_mask is None:
            return None
        block_mask_1 = np.concatenate([np.array([True], dtype=bool), self.block_mask])
        block_mask_2 = np.concatenate([self.block_mask, np.array([True], dtype=bool)])
        block_mask = np.logical_and(block_mask_1, block_mask_2)
        num_configs = len(self.npz_dict["config_runtime"])
        config_mask = np.zeros((num_configs,), dtype=bool)
        for i_block, block_ok in enumerate(block_mask):
            block_slice = slice(i_block*self.block_size, (i_block+1)*self.block_size)
            config_mask[block_slice] = block_ok
        new_dict: Dict[str, np.ndarray] = dict()
        for k, v in self.npz_dict.items():
            if k == "config_runtime":
                new_dict[k] = self.npz_dict[k][config_mask]
            elif k == "node_config_feat":
                new_dict[k] = self.npz_dict[k][config_mask]
            else:
                new_dict[k] = v
        return new_dict
