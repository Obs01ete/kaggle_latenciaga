import os
from pathlib import Path
from typing import Sequence, Dict, Optional, List
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser
import time
import datetime

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchmetrics.functional.regression import kendall_rank_corrcoef
from torchmetrics import MeanAbsolutePercentageError

from src.data import LayoutData
from src.model import Model
from src.metrics import tile_topk_metric
from src.wandb_support import init_wandb, try_upload_artefacts
from src.stats_keeper import StatsKeeper
from src.sys_utils import worker_init_fn


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
                         exclude_patterns: Optional[List[str]] = None
                         ) -> List[Dict[str, List[torch.Tensor]]]:
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if ".bias" not in name]
    if exclude_patterns is not None:
        decay_parameters = [name for name in decay_parameters
                            if not any(pat in name for pat in exclude_patterns)]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


class Trainer:
    def __init__(self,
                 source_data_path: Optional[str] = None,
                 max_iterations: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 microbatch_size: Optional[int] = None,
                 val_batch_size: Optional[int] = None,
                 oversample_factor: Optional[int] = None,
                 weight_decay: Optional[float] = None,
                 collection: Optional[str] = None,
                 tag: Optional[str] = None,
                 debug: bool = False):

        self.tag = tag
        self.debug = debug

        if collection is None:
            self.collection = "layout-xla-random"
        else:
            self.collection = collection
        print(f"{self.collection=}")

        self.is_tile = "tile-" in self.collection

        DEFAULT_DATA_PATH = "/home/khizbud/latenciaga/data/npz_all/npz"
        DEFAULT_MAX_ITERATIONS = 400_000 if self.is_tile else 200_000
        DEFAULT_BATCH_SIZE = 100 if self.is_tile else 10
        DEFAULT_MICROBATCH_SIZE = 10 if self.is_tile else 4
        DEFAULT_VAL_BATCH_SIZE = 400 if self.is_tile else 40
        DEFAULT_OVERSAMPLE_FACTOR = 100
        DEFAULT_WEIGHT_DECAY = 0.0
        DEFAULT_ITERS_PER_VAL = 10_000 if self.is_tile else 2_000

        if source_data_path is None:
            self.source_data_path = DEFAULT_DATA_PATH
        else:
            self.source_data_path = source_data_path
    
        data_root = Path(self.source_data_path).expanduser()

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

        self.iters_per_val = DEFAULT_ITERS_PER_VAL

        self.train_data = LayoutData(
            data_root,
            coll=self.collection,
            split="train",
            microbatch_size=self.microbatch_size,
            oversample_factor=self.oversample_factor)

        self.val_data = LayoutData(
            data_root,
            coll=self.collection,
            split="valid")

        self.test_data = LayoutData(
            data_root,
            coll=self.collection,
            split="test")

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

        self.artefact_dir = None
        self.logger = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device=", self.device)

        # microbatches in batch. real batch is batch_size*microbatch_size
        if batch_size is None:
            self.batch_size = DEFAULT_BATCH_SIZE
        else:
            self.batch_size = batch_size

        self.model = Model(self.is_tile)
        self.model.to(self.device)
        
        if val_batch_size is None:
            self.val_batch_size = DEFAULT_VAL_BATCH_SIZE
        else:
            self.val_batch_size = val_batch_size
        self.test_batch_size = self.val_batch_size
        self.stats = StatsKeeper()

        self.loss_op = MeanAbsolutePercentageError().to(self.device)
    
    @property
    def full_batch_size(self):
        return self.batch_size * self.microbatch_size

    def train(self):

        print("Start training")

        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        art_dir_name = (f"{datetime_str}_{self.collection}" +
                        (f"_{self.tag}" if self.tag is not None else ""))
        self.artefact_dir = os.path.join("runs", art_dir_name)
        os.makedirs(self.artefact_dir, exist_ok=True)

        if self.logger is None:
            self.logger = SummaryWriter(self.artefact_dir, flush_secs=30)

        self.iteration = 0

        # self.validate() # for quick debug

        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.worker_threads,
            pin_memory=True,
            collate_fn=concat_premade_microbatches,
            worker_init_fn=worker_init_fn)
        
        print(f"{len(train_loader)=}")

        optimized_parameters = get_model_parameters(
            self.model,
            weight_decay=self.weight_decay,
            exclude_patterns=["output_shaping"])
        
        optimizer = torch.optim.Adam(optimized_parameters, lr=1e-3)

        epoch = 0

        print(f"{torch.get_num_threads()=}")
        print("DataLoader num worker threads", self.worker_threads)

        # batch.config_runtime.mean()=2.4 for xla
        # batch.config_runtime.mean()=0.03 for nlp
        # batch.config_runtime.mean()=0.015 for tile

        # good ranking_margin = batch.config_runtime.mean / 200

        ranking_margin: float
        if "-xla-" in self.collection:
            ranking_margin = 1e-2 # sec
        elif "-nlp-" in self.collection:
            ranking_margin = 1.5e-4 # sec
        else:
            ranking_margin = 1e-1 # fractional units (not seconds)

        exit_training = False
        while True:
            if exit_training:
                break

            end_iter_ts = time.time()
            for batch in train_loader:
                data_load_ts = time.time()
                data_load_dur = data_load_ts - end_iter_ts

                train_print_interval = 10
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

                if self.is_tile:
                    loss_mape = torch.zeros(size=(1,), dtype=pred.dtype, device=pred.device)
                else:
                    loss_mape = self.loss_op(pred, batch.config_runtime)

                # We store #ub_size duplicates of diff_triu_vector
                # because of the technicalities of batching.
                # In actuality we need only one of them for microbatch.
                diff_triu_vector_per_ub = \
                    batch.diff_triu_vector[::self.microbatch_size]

                if self.is_tile:
                    scale = 1.0
                else:
                    scale = 1.0 / ranking_margin

                loss_diff_mat = scale * F.margin_ranking_loss(
                    pred_diff_mat,
                    torch.zeros_like(pred_diff_mat),
                    torch.sign(diff_triu_vector_per_ub),
                    margin=ranking_margin,
                    reduce=False)

                nz_diff_loss_frac = ((loss_diff_mat > 1e-6).sum() /
                                          loss_diff_mat.numel())

                loss_diff_mat_red = torch.mean(loss_diff_mat)

                diff_mat_loss_scale = 1.0
                loss_diff_mat_sc = diff_mat_loss_scale * loss_diff_mat_red
                loss = loss_mape + loss_diff_mat_sc

                if self.iteration % train_print_interval == 0:
                    loss_val = loss.item()
                    loss_mape_val = loss_mape.item()
                    loss_diff_mat_sc_val = loss_diff_mat_sc.item()
                    nz_diff_loss_frac = nz_diff_loss_frac.item()
                    print(f"Train loss = {loss_val:.5f}, "
                        f"loss_mape = {loss_mape_val:.5f}, "
                        f"loss_diff_mat_sc = {loss_diff_mat_sc_val:.5f}, "
                        f"nz_diff_loss_frac = {nz_diff_loss_frac:.5f}"
                        )
                    self.logger.add_scalar("train/loss", loss_val, self.iteration)
                    self.logger.add_scalar("train/loss_mape", loss_mape, self.iteration)
                    self.logger.add_scalar("train/loss_diff_mat_sc", loss_diff_mat_sc, self.iteration)
                    self.logger.add_scalar("train/nz_diff_loss_frac", nz_diff_loss_frac, self.iteration)

                    kendall_list = []
                    p_value_list = []
                    for iub in range(self.batch_size):
                        ub_slice = slice(iub*self.microbatch_size, (iub+1)*self.microbatch_size)
                        kendall, p_value = kendall_rank_corrcoef(
                            pred[ub_slice], batch.config_runtime[ub_slice],
                            t_test=True, alternative='two-sided')
                        kendall_list.append(kendall)
                        p_value_list.append(p_value)
                    kendall_total = torch.nanmean(torch.stack(kendall_list)).item()
                    p_value_total = torch.nanmean(torch.stack(p_value_list)).item()
                    self.stats.update_train(
                        iteration=self.iteration,
                        train_kendall=kendall_total,
                        train_loss=loss_val,
                        train_mape=loss_mape_val,
                        train_loss_diff_mat_sc=loss_diff_mat_sc_val,
                        train_nz_diff_loss_frac=nz_diff_loss_frac,
                    )
                    print("kendall=", kendall_total, "p_value=", p_value_total)
                    self.logger.add_scalar("train/kendall", kendall_total, self.iteration)
                    self.logger.add_scalar("train/p_value", p_value_total, self.iteration)
                    self.logger.add_scalar("train/epoch", epoch, self.iteration)

                optimizer.zero_grad()
                loss.backward()
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

                end_iter_ts = time.time()

                self.iteration += 1
                if self.iteration >= self.max_iterations:
                    exit_training = True
                    break
            
            epoch += (self.oversample_factor
                      if self.oversample_factor is not None else 1)

        self._save_ckpt()
        self.test("end_of_train_submission.csv")
    
    def validate(self):
        print("Validating...")
        self._validate('valid', "submission_val_auto.csv")
        self._validate('test', "submission_test_auto.csv")
        print("Validation done")
    
    def test(self, submission_csv_path: str):
        print("Testing...")
        self._validate('test', submission_csv_path)
        self._validate('valid', submission_csv_path+"_val.csv")
        print("Testing done")

    def _validate(self, split: str,
                  submission_csv_path: Optional[str] = None):

        assert split in {'valid', 'test'}

        loss_list, prediction_dict = self._make_predictions(split)
        
        if split == 'valid':
            val_kendall, val_loss = self._compute_metrics(
                split, loss_list, prediction_dict)
            self.stats.update_val(
                iteration=self.iteration,
                val_kendall=val_kendall,
                val_loss=val_loss,
            )

        if submission_csv_path is not None:
            self._prepare_submission(prediction_dict, submission_csv_path)

    def _prepare_submission(self,
                            prediction_dict: Dict[str, Dict[str, List[float]]],
                            submission_csv_path: str):
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

    def _save_ckpt(self, ckpt_name="latest.pth", best_ckpt_name="best.pth"):
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
        loader = DataLoader(
            self.val_data if split == 'valid' else self.test_data,
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
                    loss = self.loss_op(pred, batch.config_runtime)
                    loss_list.append(loss.item())

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
                         prediction_dict: Dict[str, Dict[str, List[float]]]):
        if loss_list is not None and len(loss_list) > 0:
            loss_grand = np.mean(np.array(loss_list, dtype=np.float32)).item()
            print(f"{split} loss = {loss_grand:.5f}")
        else:
            loss_grand = float('nan')

        kendall_grand_list = []
        p_value_grand_list = []
        tile_topk_list = []
        for name, dol in prediction_dict.items():
            pred_all = np.array(dol['pred_list'], dtype=np.float32)
            target_all = np.array(dol['target_list'], dtype=np.float32)

            kendall, p_value = kendall_rank_corrcoef(
                torch.tensor(pred_all), torch.tensor(target_all),
                t_test=True, alternative='two-sided')

            kendall_grand_list.append(kendall.item())
            p_value_grand_list.append(p_value.item())
            if self.is_tile:
                tile_metric = tile_topk_metric(torch.tensor(pred_all), torch.tensor(target_all))
                tile_topk_list.append(tile_metric.item())

        kendall_grand = np.mean(kendall_grand_list).item()
        p_value_grand = np.mean(p_value_grand_list).item()
        print(f"{split} kendall=", kendall_grand, "p_value=", p_value_grand)
        if self.is_tile:
            tile_topk_grand = np.mean(tile_topk_list).item()
            print(f"{split} tile_top_k=", tile_topk_grand)
            self.logger.add_scalar("val/tile_top_k", tile_topk_grand, self.iteration)
        if self.logger is not None:
            self.logger.add_scalar("val/loss", loss_grand, self.iteration)
            self.logger.add_scalar("val/kendall", kendall_grand, self.iteration)
            self.logger.add_scalar("val/p_value", p_value_grand, self.iteration)
        return kendall_grand, loss_grand

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
        loader = DataLoader(
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
    parser.add_argument('--source-data-path', action='store', type=str,
                        help='Provide path to data folder in format */data/npz_all/npz')
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
    parser.add_argument('--test-snapshot', action='store', type=str,
                        help='Provide .pth, get submission.csv')
    parser.add_argument('--start-from-pth', action='store', type=str,
                        help='In training mode, start with the provided .pth')
    parser.add_argument('--test-val-submission-csv', action='store', type=str,
                        help='Provide submission_val.csv, get score')
    parser.add_argument('--collection', action='store', type=str,
                        help='One of 4 collections. Default layout-xla-random if not set')
    parser.add_argument('--tag', action='store', type=str,
                        help='Extra suffix to put on the artefact dir name')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--enable-wandb', action='store_true')
    args = parser.parse_args()

    trainer = Trainer(source_data_path=args.source_data_path,
                      max_iterations=args.max_iterations,
                      batch_size=args.batch_size,
                      microbatch_size=args.microbatch_size,
                      val_batch_size=args.val_batch_size,
                      oversample_factor=args.oversample_factor,
                      weight_decay=args.weight_decay,
                      collection=args.collection,
                      tag=args.tag,
                      debug=args.debug)
    if args.test_snapshot is not None:
        trainer.load_snapshot(args.test_snapshot)
        trainer.test("test_submission.csv")
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
