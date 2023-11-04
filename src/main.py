import os
from pathlib import Path
from typing import Sequence, Dict, Optional, List, Any, Tuple, Callable
from click import Option
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque
from functools import partial
from argparse import ArgumentParser
import time
import datetime
import math
import pandas as pd

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils.clip_grad import clip_grad_norm_

from torchmetrics.functional.regression import kendall_rank_corrcoef
from torchmetrics import MeanAbsolutePercentageError

from src.data import LayoutData, Purpose
from src.model import Model
from src.metrics import tile_topk_metric
from src.wandb_support import init_wandb, try_upload_artefacts
from src.stats_keeper import StatsKeeper
from src.sys_utils import worker_init_fn

from src.allrank_losses.listMLE import listMLE


def concat_premade_microbatches(microbatch_list: Sequence[Batch]):
    grand_list = []
    for micribatch in microbatch_list:
        for sample in micribatch.to_data_list():
            grand_list.append(sample)
    batch = Batch.from_data_list(grand_list, follow_batch=LayoutData.FOLLOW_BATCH_KEYS)
    return batch


# from transformers/src/transformers/trainer_pt_utils.py
def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_model_parameters(model: torch.nn.Module,
                         weight_decay: float = 0,
                         explicit_assignment: Optional[Dict[str, float]] = None,
                         exclude_patterns: Optional[List[str]] = None
                         ) -> List[Dict[str, Any]]:
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    if explicit_assignment is not None:
        decay_parameters = [name for name in decay_parameters
                            if name not in explicit_assignment.keys()]
    decay_parameters = [name for name in decay_parameters if ".bias" not in name]
    if exclude_patterns is not None:
        decay_parameters = [name for name in decay_parameters
                            if not any(pat in name for pat in exclude_patterns)]
    explicit_assignment_keys = explicit_assignment.keys() \
        if explicit_assignment is not None else set()
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if n not in decay_parameters and
                       n not in explicit_assignment_keys],
            "weight_decay": 0.0,
        },
    ]
    if explicit_assignment is not None:
        for name, decay in explicit_assignment.items():
            group = dict(params=[p for n, p in model.named_parameters() if n == name],
                        weight_decay=decay)
            optimizer_grouped_parameters.append(group)
    return optimizer_grouped_parameters


def insert_suffix(path: str, suffix: str) -> str:
    path_wo_ext, ext = os.path.splitext(path)
    return path_wo_ext + suffix + ext


def L2Clip(pred: torch.Tensor, max_norm: float, dim: int = -1):
    norms = torch.norm(pred, dim=dim)
    overshoots = torch.maximum(norms, max_norm*torch.ones_like(norms)) / \
        (max_norm*torch.ones_like(norms))
    clipped = pred / overshoots.unsqueeze(1)
    return clipped


class Trainer:
    def __init__(self,
                 data_root: Optional[str] = None,
                 max_iterations: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 microbatch_size: Optional[int] = None,
                 val_batch_size: Optional[int] = None,
                 oversample_factor: Optional[int] = None,
                 weight_decay: Optional[float] = None,
                 wider_config: Optional[bool] = None,
                 collection: Optional[str] = None,
                 delete_duplicates: Optional[bool] = None,
                 enable_trainval: Optional[bool] = None,
                 validate_first: Optional[bool] = None,
                 tag: Optional[str] = None,
                 debug: bool = False) -> None:

        self.tag = tag
        self.debug = debug

        if collection is None:
            self.collection = "layout-xla-random"
        else:
            self.collection = collection
        print(f"{self.collection=}")

        if delete_duplicates is None:
            delete_duplicates = True # turn on duplicate filtration by default

        if enable_trainval is None:
            enable_trainval = False

        if validate_first is None:
            self.validate_first = False
        else:
            self.validate_first = validate_first

        self.is_tile = "tile-" in self.collection
        self.is_nlp = "-nlp-" in self.collection
        self.is_default = "-default" in self.collection

        DEFAULT_DATA_ROOT = "/home/khizbud/latenciaga/data"
        DEFAULT_MAX_ITERATIONS = 400_000 if self.is_tile else 200_000
        DEFAULT_BATCH_SIZE = 100 if self.is_tile else 4 if self.is_nlp else 4
        DEFAULT_MICROBATCH_SIZE = 10 if self.is_tile else 10 if self.is_nlp else 10
        DEFAULT_VAL_BATCH_SIZE = 400 if self.is_tile else 40
        DEFAULT_OVERSAMPLE_FACTOR = 100
        DEFAULT_WEIGHT_DECAY = 0.0
        DEFAULT_ITERS_PER_VAL = 10_000 if self.is_tile else 2_000
        DEFAULT_ITERS_PER_TRAIN_KENDALL_PRINT = 2_500 if self.is_tile else 500

        if data_root is None:
            self.data_root = DEFAULT_DATA_ROOT
        else:
            self.data_root = data_root

        collections_root_str = os.path.join(self.data_root, "npz_all/npz")
        collections_root = Path(collections_root_str).expanduser()

        if microbatch_size is None:
            self.microbatch_size = DEFAULT_MICROBATCH_SIZE
        else:
            self.microbatch_size = microbatch_size

        if oversample_factor is None:
            self.oversample_factor = DEFAULT_OVERSAMPLE_FACTOR
        else:
            self.oversample_factor = oversample_factor

        if weight_decay is not None:
            self.weight_decay = weight_decay
        else:
            self.weight_decay = DEFAULT_WEIGHT_DECAY

        if wider_config is None:
            wider_config = False

        self.iters_per_val = DEFAULT_ITERS_PER_VAL
        self.iters_per_train_kendall_print = DEFAULT_ITERS_PER_TRAIN_KENDALL_PRINT

        cache_root = Path("data_cache_clean") if delete_duplicates else Path("data_cache")

        self.train_data = LayoutData(
            collections_root,
            coll=self.collection,
            split="trainval" if enable_trainval else "train",
            purpose=Purpose.Train,
            microbatch_size=self.microbatch_size,
            oversample_factor=self.oversample_factor,
            cache_root=cache_root,
            delete_duplicates=delete_duplicates,
        )

        self.val_data = LayoutData(
            collections_root,
            coll=self.collection,
            split="valid",
            purpose=Purpose.Valid,
            cache_root=cache_root,
            delete_duplicates=delete_duplicates,
        )

        # don't delete duplicates for test but use the same folder as train and val
        self.test_data = LayoutData(
            collections_root,
            coll=self.collection,
            cache_root=cache_root,
            delete_duplicates=False,
            split="test",
            purpose=Purpose.Test)

        if max_iterations is None:
            self.max_iterations = DEFAULT_MAX_ITERATIONS
        else:
            self.max_iterations = max_iterations
        
        default_worker_threads = 16

        cpu_count = os.cpu_count()
        if cpu_count is not None:
            worker_threads = min(cpu_count, default_worker_threads)
        else:
            worker_threads = default_worker_threads
        self.worker_threads = 0 if self.debug else worker_threads

        self.art_dir_name: Optional[str] = None
        self.artefact_dir: Optional[str] = None
        self.logger: Optional[SummaryWriter] = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device=", self.device)

        # microbatches in batch. real batch is batch_size*microbatch_size
        if batch_size is None:
            self.batch_size = DEFAULT_BATCH_SIZE
        else:
            self.batch_size = batch_size

        self.model = Model(self.is_tile, self.is_nlp, self.is_default,
                           wider_config)
        self.model.to(self.device)
        
        if val_batch_size is None:
            self.val_batch_size = DEFAULT_VAL_BATCH_SIZE
        else:
            self.val_batch_size = val_batch_size
        self.test_batch_size = self.val_batch_size
        self.stats = StatsKeeper()

        self.iteration: Optional[int] = None
    
    @property
    def full_batch_size(self):
        return self.batch_size * self.microbatch_size

    @property
    def is_xla(self) -> bool:
        return not self.is_nlp

    @property
    def is_random(self) -> bool:
        return not self.is_default

    @property
    def is_layout(self) -> bool:
        return not self.is_tile

    def train(self) -> None:

        print("Start training")

        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.art_dir_name = (f"{datetime_str}_{self.collection}" +
                        (f"_{self.tag}" if self.tag is not None else ""))
        self.artefact_dir = os.path.join("runs", self.art_dir_name)
        os.makedirs(self.artefact_dir, exist_ok=True)

        if self.logger is None:
            self.logger = SummaryWriter(self.artefact_dir, flush_secs=30)

        self.iteration = 0

        if self.validate_first:
            self.validate()

        train_loader: DataLoader[Batch] = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.worker_threads,
            pin_memory=True,
            collate_fn=concat_premade_microbatches,
            worker_init_fn=worker_init_fn)
        
        print(f"{len(train_loader)=}")

        output_shaping_wd = 1e-4
        optimized_parameters = get_model_parameters(
            self.model,
            weight_decay=self.weight_decay,
            explicit_assignment={
                'output_shaping.weight': output_shaping_wd,
                'output_shaping.bias': output_shaping_wd})
        
        optimizer = torch.optim.Adam(optimized_parameters, lr=1e-3, eps=1e-6)

        # milestones = [v*10_000 for v in (1, 2, 3, 4)] # finalization schedule for layout
        # milestones = [v*20_000 for v in (1, 2, 3, 4)] # finalization schedule for tile
        milestones = [int(self.max_iterations*v) for v in (0.6, 0.7, 0.8, 0.9)]
        lr_scheduler = MultiStepLR(optimizer,
                                   milestones=milestones,
                                   gamma=1/math.sqrt(10))

        epoch = 0

        print(f"{torch.get_num_threads()=}")
        print("DataLoader num worker threads", self.worker_threads)

        bound_deque: Callable[[], deque[float]] = partial(deque, maxlen=100)
        kendall_deq_dict: Dict[str, deque[float]] = defaultdict(bound_deque)

        # We need gradient accumulation to improve stability of training
        # Without gradient accumulation the number of microbatches (slates)
        # is too small (4) and leads to gradient explosion. We need to accumulate
        # something like 4 iterations to the total of 16 microbatches.
        # Earlier we had 10 microbatches and the training was fine.
        NUM_ACCUM_STEPS: int = 1 # 4

        exit_training = False
        while True:
            if exit_training:
                break

            end_iter_ts = time.time()
            for batch in train_loader:
                data_load_ts = time.time()
                data_load_dur = data_load_ts - end_iter_ts

                train_print_interval = 100
                if self.iteration % train_print_interval == 0:
                    print("-"*80)
                    print(self.iteration, batch)

                batch = batch.to(self.device, non_blocking=True)

                self.model.train()

                pred, pred_diff_mat = self.model(
                    batch.node_feat,
                    batch.node_opcode,
                    batch.batch,
                    batch.ptr,

                    batch.node_config_feat if not self.is_tile else None,
                    batch.node_config_ids if not self.is_tile else None,
                    batch.node_config_feat_ptr if not self.is_tile else None,

                    batch.config_feat if self.is_tile else None,
                    batch.config_feat_ptr if self.is_tile else None,

                    batch.edge_index,
                    
                    self.microbatch_size)

                target = batch.config_runtime
                target_slated = target.view(-1, self.microbatch_size)
                pred_slated_raw: torch.Tensor = pred.view(-1, self.microbatch_size)
                pred_slated = L2Clip(pred_slated_raw, max_norm=500, dim=-1)

                loss = listMLE(pred_slated, target_slated)

                if self.iteration % train_print_interval == 0:
                    loss_val = loss.item()
                    learning_rate = lr_scheduler.get_last_lr()[0]
                    print(f"Train loss = {loss_val:.5f}, "
                        )
                    self.logger.add_scalar("train/loss", loss_val, self.iteration)
                    self.logger.add_scalar("train/learning_rate", learning_rate, self.iteration)

                    kendall_list = []
                    p_value_list = []
                    for iub in range(self.batch_size):
                        ub_slice = slice(iub*self.microbatch_size, (iub+1)*self.microbatch_size)
                        try:
                            kendall, p_value = kendall_rank_corrcoef(
                                pred[ub_slice], batch.config_runtime[ub_slice],
                                t_test=True, alternative='two-sided')
                        except NotImplementedError as ex:
                            print(f"Warning: Skipping exception {str(ex)}")
                            kendall = torch.tensor([float('nan')], dtype=pred.dtype, device=pred.device)
                            p_value = torch.tensor([float('nan')], dtype=pred.dtype, device=pred.device)
                        kendall_list.append(kendall)
                        p_value_list.append(p_value)

                    # keep stats per train file
                    for fname, kendall_val in zip(
                            batch.fname[::self.microbatch_size],
                            [v.item() for v in kendall_list]):
                        kendall_deq_dict[fname].append(kendall_val)

                    kendall_total = torch.nanmean(torch.stack(kendall_list)).item()
                    p_value_total = torch.nanmean(torch.stack(p_value_list)).item()
                    self.stats.update_train(
                        iteration=self.iteration,
                        train_kendall=kendall_total,
                        train_loss=loss_val,
                        train_mape=0.0,
                        train_loss_diff_mat_sc=0.0,
                        train_nz_diff_loss_frac=0.0,
                    )
                    print("kendall=", kendall_total, "p_value=", p_value_total)
                    self.logger.add_scalar("train/kendall", kendall_total, self.iteration)
                    # self.logger.add_scalar("train/p_value", p_value_total, self.iteration)
                    self.logger.add_scalar("train/epoch", epoch, self.iteration)

                    all_param_norm = [torch.norm(p.detach()) for p in self.model.parameters()]
                    total_param_norm = torch.norm(torch.tensor(all_param_norm)).item()
                    self.logger.add_scalar(f"train/param_norm", total_param_norm, self.iteration)

                    mean_pred_raw = torch.mean(torch.abs(pred_slated_raw)).item()
                    mean_pred = torch.mean(torch.abs(pred_slated)).item()
                    mean_target = torch.mean(torch.abs(target_slated)).item()
                    self.logger.add_scalar(f"train/mean_pred_raw", mean_pred_raw, self.iteration)
                    self.logger.add_scalar(f"train/mean_pred", mean_pred, self.iteration)
                    self.logger.add_scalar(f"train/mean_target", mean_target, self.iteration)

                grad_accum_cnt = self.iteration % NUM_ACCUM_STEPS
                if grad_accum_cnt % NUM_ACCUM_STEPS == 0:
                    optimizer.zero_grad()
                loss.backward(retain_graph=True)
                if grad_accum_cnt == NUM_ACCUM_STEPS - 1:
                    grad_clip_val = 100.0
                    grad_norm = clip_grad_norm_(self.model.parameters(), grad_clip_val).item()
                    loss_print_interval = train_print_interval * NUM_ACCUM_STEPS
                    if self.iteration % loss_print_interval == loss_print_interval - 1:
                        self.logger.add_scalar(f"train/loss_grad_norm", grad_norm, self.iteration)
                    optimizer.step()

                forward_backward_ts = time.time()
                forward_backward_dur = forward_backward_ts - data_load_ts

                if self.iteration % train_print_interval == 0:
                    print(f"{data_load_dur=}, {forward_backward_dur=}")
                    self.logger.add_scalar("train/data_load_dur", data_load_dur, self.iteration)
                    self.logger.add_scalar("train/forward_backward_dur", forward_backward_dur, self.iteration)

                if self.iteration % self.iters_per_val == self.iters_per_val - 1:
                    self.validate()
                    self._save_ckpt()
                    validation_ts = time.time()
                    validation_dur = validation_ts - forward_backward_ts
                    print(f"{validation_dur=}")
                    self.logger.add_scalar("val/validation_dur", validation_dur, self.iteration)

                if self.iteration % self.iters_per_train_kendall_print == 0:
                    self._log_train_kendall(kendall_deq_dict)

                end_iter_ts = time.time()

                lr_scheduler.step()
                self.iteration += 1
                if self.iteration >= self.max_iterations:
                    exit_training = True
                    break
            
            epoch += (self.oversample_factor
                      if self.oversample_factor is not None else 1)

        self._save_ckpt()
        self.test("end_of_train_sub.csv")

    def _log_train_kendall(self, kendall_deq_dict: Dict[str, deque[float]]):
        assert self.logger is not None
        kendall_agg_dict = {}
        for k in kendall_deq_dict:
            kendall_agg_dict[k] = np.mean(list(kendall_deq_dict[k]))
        sorted_by_name_dict = dict(sorted(kendall_agg_dict.items()))
        sorted_by_value_dict = dict(sorted(kendall_agg_dict.items(),
                                            key=lambda tup: tup[1]))
        sorted_by_name_str = "  \n".join([f"{k}: {v}"
                                            for k, v in sorted_by_name_dict.items()])
        sorted_by_value_str = "  \n".join([f"{k}: {v}"
                                            for k, v in sorted_by_value_dict.items()])
        if self.logger:
            self.logger.add_text(
                "train kendall (sorted by name)",
                sorted_by_name_str,
                self.iteration
            )
            self.logger.add_text(
                "train kendall (sorted by kendall)",
                sorted_by_value_str,
                self.iteration
            )
        return
    
    def validate(self):
        print("Validating...")
        auto_name = "auto.csv"
        self._validate('valid', insert_suffix(auto_name, "_val"))
        self._validate('test', auto_name)
        print("Validation done")
    
    def test(self, submission_csv_path: str):
        print("Testing...")
        self._validate('test', submission_csv_path)
        self._validate('valid', insert_suffix(submission_csv_path, "_val"))
        print("Testing done")

    def _validate(self, split: str,
                  submission_csv_path: Optional[str] = None):

        assert split in {'valid', 'test'}

        iteration = self.iteration if self.iteration is not None else 0

        loss_list, prediction_dict = self._make_predictions(split)

        kendall_dict = None
        if split in {'valid'}:
            val_kendall, val_loss, kendall_dict = self._compute_metrics(
                split, loss_list, prediction_dict)
            self.stats.update_val(
                iteration=iteration,
                val_kendall=val_kendall,
                val_loss=val_loss,
            )

        if submission_csv_path is not None:
            self._prepare_submission(prediction_dict, submission_csv_path,
                  make_injected = split == 'test')

            if kendall_dict is not None:
                with open(insert_suffix(submission_csv_path, "_details"), "w") as f:
                    for key, val in kendall_dict.items():
                        f.write(f"{key},{val}\n")

        torch.cuda.empty_cache()

    def _prepare_submission(self,
                            prediction_dict: Dict[str, Dict[str, List[float]]],
                            submission_csv_path: str,
                            make_injected: bool = False):
        # Prepare submission
        submission_dict: Dict[str, np.ndarray] = dict()
        for name, dol in prediction_dict.items():
            pred_all = np.array(dol['pred_list'], dtype=np.float32)
            # NB: pred_all are guaranteed to be sequential by DataLoader
            submission_ranks = np.argsort(pred_all)
            if self.is_tile:
                submission_ranks = submission_ranks[:5]
            submission_dict[name] = submission_ranks

        if self.artefact_dir is not None:
            submission_csv_path = os.path.join(self.artefact_dir,
                                               submission_csv_path)
        self.save_submission(submission_dict, submission_csv_path)
        print(f"Saved to {submission_csv_path}")

        if make_injected:
            sub = pd.read_csv(os.path.join(self.data_root, 'sample_submission.csv'))
            orig_len = len(sub)
            for filename, ranks_np in submission_dict.items():
                id = self.collection.replace('-', ':') + ':' + filename
                ranks_str = ';'.join([str(int(v)) for v in ranks_np])
                sub.loc[sub.ID == id, 'TopConfigs'] = ranks_str
            if len(sub) != orig_len:
                print("WARNING: injected submission may be corrupted")
            extra_suffix = f"_{self.art_dir_name}" if self.art_dir_name is not None else ""
            injected_path = insert_suffix(submission_csv_path, f"_injected{extra_suffix}")
            sub.to_csv(injected_path, index=False)
            print(f"Saved to {injected_path}")

        return

    def _save_ckpt(self, ckpt_name="latest.pth", best_ckpt_name="best.pth"):
        assert self.artefact_dir is not None
        torch.save(self.model.state_dict(),
                   os.path.join(self.artefact_dir, ckpt_name))
        self.stats.save_as_json(os.path.join(self.artefact_dir,
                                             f"{ckpt_name.split('.')[0]}.json"))

        if self.stats.best_val_kendall == self.stats.val_kendall:
            torch.save(self.model.state_dict(),
                       os.path.join(self.artefact_dir, best_ckpt_name))
            self.stats.save_as_json(os.path.join(self.artefact_dir,
                                                 f"{best_ckpt_name.split('.')[0]}.json"))

        try_upload_artefacts(self.artefact_dir)

    def _make_predictions(self, split: str):
        worker_threads = (self.worker_threads if self.is_tile
                          else self.worker_threads // 4)
        data = self.val_data if split == 'valid' else \
            self.test_data  # else test
        loader: DataLoader[Batch] = DataLoader(
            data,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=worker_threads,
            pin_memory=True,
            collate_fn=concat_premade_microbatches,
            worker_init_fn=worker_init_fn)
        
        print("len(loader)=", len(loader))

        prediction_dict: Dict[str, Dict[str, List[float]]] = \
            defaultdict(lambda: defaultdict(list))
        loss_list = []

        for i_batch, batch in tqdm(enumerate(loader)):

            batch = batch.to(self.device, non_blocking=True)

            self.model.eval()

            with torch.no_grad():
                pred, _ = self.model(
                    batch.node_feat,
                    batch.node_opcode,
                    batch.batch,
                    batch.ptr,

                    batch.node_config_feat if not self.is_tile else None,
                    batch.node_config_ids if not self.is_tile else None,
                    batch.node_config_feat_ptr if not self.is_tile else None,

                    batch.config_feat if self.is_tile else None,
                    batch.config_feat_ptr if self.is_tile else None,

                    batch.edge_index,
                    
                    ub_size=1 # microbatch is 1 for val/test
                    )

                if split == 'valid':
                    loss_list.append(0.0)

            for pr, tg, fn in zip(pred, batch.config_runtime, batch.fname):
                assert isinstance(fn, str), "File name must be a string"
                dol = prediction_dict[fn]
                dol['pred_list'].append(pr.item())
                dol['target_list'].append(tg.item())

            if self.debug:
            # if True:
                if i_batch >= 2:
                    break
        
        return loss_list, prediction_dict
    
    def _compute_metrics(self, split:str,
                         loss_list: Optional[List[float]],
                         prediction_dict: Dict[str, Dict[str, List[float]]]) -> \
                         Tuple[float, float, Dict[str, float]]:
        if loss_list is not None and len(loss_list) > 0:
            loss_grand = np.mean(np.array(loss_list, dtype=np.float32)).item()
            print(f"{split} loss = {loss_grand:.5f}")
        else:
            loss_grand = float('nan')

        kendall_grand_list = []
        p_value_grand_list = []
        tile_topk_list = []
        kendall_dict: Dict[str, float] = {}
        for name, dol in prediction_dict.items():
            pred_all = np.array(dol['pred_list'], dtype=np.float32)
            target_all = np.array(dol['target_list'], dtype=np.float32)

            kendall, p_value = kendall_rank_corrcoef(
                torch.tensor(pred_all), torch.tensor(target_all),
                t_test=True, alternative='two-sided')

            graph_not_good = (
                (self.is_layout and "unet" in name) or
                (self.collection == "layout-xla-default" and "mlperf_bert_batch_24_2x2" in name)
            )
            if not graph_not_good:
                kendall_grand_list.append(kendall.item())
                p_value_grand_list.append(p_value.item())

            if self.is_tile:
                tile_metric = tile_topk_metric(torch.tensor(pred_all), torch.tensor(target_all))
                tile_topk_list.append(tile_metric.item())

            if self.logger is not None:
                self.logger.add_scalar(f"val/kendall/{name}", kendall.item(), self.iteration)

            print(f"val/kendall/{name} {kendall.item()}")
            kendall_dict[name] = kendall.item()

        kendall_grand = np.mean(kendall_grand_list).item()
        p_value_grand = np.mean(p_value_grand_list).item()
        print(f"{split} kendall=", kendall_grand, "p_value=", p_value_grand)

        if self.is_tile:
            tile_topk_grand = np.mean(tile_topk_list).item()
            print(f"{split} tile_top_k=", tile_topk_grand)
            if self.logger is not None:
                self.logger.add_scalar("val/tile_top_k", tile_topk_grand, self.iteration)

        if self.logger is not None:
            self.logger.add_scalar("val/loss", loss_grand, self.iteration)
            self.logger.add_scalar("val/kendall", kendall_grand, self.iteration)
            # self.logger.add_scalar("val/p_value", p_value_grand, self.iteration)
        return kendall_grand, loss_grand, kendall_dict

    def save_submission(self,
                        submission_dict: Dict[str, np.ndarray],
                        csv_name: str):
        with open(csv_name, "w") as f:
            for npz_name, ranks in submission_dict.items():
                ranks_list = [str(v) for v in list(ranks)]
                line = self.collection.replace('-', ':') + ":" + npz_name + "," + \
                    ";".join(ranks_list) + "\n"
                f.write(line)
    
    def load_snapshot(self, snapshot_path: str):
        self.model.load_state_dict(torch.load(snapshot_path,
                                              map_location=self.device))
    
    def test_val_submission_csv(self, submission_csv_path: str):
        loader: DataLoader[Batch] = DataLoader(
            self.val_data,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.worker_threads,
            pin_memory=True,
            collate_fn=concat_premade_microbatches,
            worker_init_fn=worker_init_fn)

        print("len(loader)=", len(loader))

        prediction_dict: Dict[str, Dict[str, List[float]]] = \
            defaultdict(lambda: defaultdict(list))

        for _, batch in tqdm(enumerate(loader)):

            for tg, fn in zip(batch.config_runtime, batch.fname):
                assert isinstance(fn, str), "File name must be a string"
                dol = prediction_dict[fn]
                dol['target_list'].append(tg.item())

        self._compute_metrics('valid', None, prediction_dict)


def main():
    parser = ArgumentParser(description='Latenciaga')
    parser.add_argument('--data-root', action='store', type=str,
                        help='Provide path to data folder that holds npz_all/ and sample_submission.csv')
    parser.add_argument('--max-iterations', '-i', action='store', type=int,
                        help='Maximum number of iterations')
    parser.add_argument('--batch-size', action='store', type=int,
                        help='Batch size')
    parser.add_argument('--microbatch-size', action='store', type=int,
                        help='Micro batch size')
    parser.add_argument('--val-batch-size', action='store', type=int,
                        help='Validation batch size')
    parser.add_argument('--oversample-factor', action='store', type=int,
                        help='Oversample factor')
    parser.add_argument('--weight-decay', action='store', type=float,
                        help='Weight decay')
    parser.add_argument('--wider-config', action='store_true')
    parser.add_argument('--test-snapshot', action='store', type=str,
                        help='Provide .pth, get submission.csv')
    parser.add_argument('--start-from-pth', action='store', type=str,
                        help='In training mode, start with the provided .pth')
    parser.add_argument('--test-val-submission-csv', action='store', type=str,
                        help='Provide submission_val.csv, get score')
    parser.add_argument('--collection', action='store', type=str,
                        help='One of 4 collections. Default layout-xla-random if not set')
    parser.add_argument('--delete-duplicates', action='store', type=bool,
                        help='Delete duplicates from source data')
    parser.add_argument('--enable-trainval', action='store', type=bool,
                        help='Enable training on merged train and valid')
    parser.add_argument('--validate-first', action='store_true',
                        help="Enable to run validation before training starts")
    parser.add_argument('--tag', action='store', type=str,
                        help='Extra suffix to put on the artefact dir name')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--enable-wandb', action='store_true')
    args = parser.parse_args()

    trainer = Trainer(data_root=args.data_root,
                      max_iterations=args.max_iterations,
                      batch_size=args.batch_size,
                      microbatch_size=args.microbatch_size,
                      val_batch_size=args.val_batch_size,
                      oversample_factor=args.oversample_factor,
                      weight_decay=args.weight_decay,
                      wider_config=args.wider_config,
                      collection=args.collection,
                      delete_duplicates=args.delete_duplicates,
                      enable_trainval=args.enable_trainval,
                      validate_first=args.validate_first,
                      tag=args.tag,
                      debug=args.debug)
    if args.test_snapshot is not None:
        trainer.load_snapshot(args.test_snapshot)
        trainer.test("test_sub.csv")
    elif args.test_val_submission_csv is not None:
        trainer.test_val_submission_csv(args.test_val_submission_csv)
    else:
        if args.start_from_pth is not None:
            trainer.load_snapshot(args.start_from_pth)
            print(f"Loading snapshot from {args.start_from_pth}")
        if args.enable_wandb:
            init_wandb(debug=args.debug)
        trainer.train()
    print("Done")


if __name__ == "__main__":
    main()
