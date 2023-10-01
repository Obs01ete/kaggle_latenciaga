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


def concat_premade_microbatches(microbatch_list: Sequence[Batch]):
    grand_list = []
    for micribatch in microbatch_list:
        for sample in micribatch.to_data_list():
            grand_list.append(sample)
    batch = Batch.from_data_list(grand_list, follow_batch=LayoutData.FOLLOW_BATCH_KEYS)
    return batch


def worker_init_fn(worker_id: int) -> None:
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        os.sched_setaffinity(0, range(cpu_count))


class Trainer:
    def __init__(self,
                 source_data_path: Optional[str] = None,
                 collection: Optional[str] = None,
                 tag: Optional[str] = None,
                 debug: bool = False):

        self.tag = tag
        self.debug = debug

        DEFAULT_DATA_PATH = "/home/khizbud/latenciaga/data/npz_all/npz"

        if source_data_path is None:
            source_data_path = DEFAULT_DATA_PATH
    
        data_root = Path(source_data_path).expanduser()

        self.microbatch_size = 4

        if collection is None:
            self.collection = "layout-xla-random"
        else:
            self.collection = collection
        print("collection=", self.collection)

        self.oversample_factor = 100 # 10 is good, crashes at 10 and 100

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
        
        self.is_tile = "tile-" in self.collection

        # microbatches in batch. real batch is batch_size*microbatch_size
        self.batch_size = 100 if self.is_tile else 10

        self.model = Model(self.is_tile)
        self.model.to(self.device)

        self.val_batch_size = 400 if self.is_tile else 40
        self.test_batch_size = self.val_batch_size

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
        
        print("len(train_loader)=", len(train_loader))
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        max_iterations = 400_000 if self.is_tile else 100_000
        epoch = 0

        print("torch.get_num_threads=", torch.get_num_threads())
        print("DataLoader num worker threads", self.worker_threads)

        # batch.config_runtime.mean()=2.4 for xla
        # batch.config_runtime.mean()=0.03 for nlp
        # batch.config_runtime.mean()=0.015 for tile

        ranking_margin: float
        if "-xla-" in self.collection:
            ranking_margin = 1e-2 # sec
        elif "-nlp-" in self.collection:
            ranking_margin = 1.5e-4 # sec
        else:
            # Need to find good margin for tile
            ranking_margin = 7.5e-5 # sec

        exit_training = False
        while True:
            if exit_training:
                break

            end_iter_ts = time.time()
            for batch in train_loader:
                data_load_ts = time.time()
                data_load_dur = data_load_ts - end_iter_ts

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

                loss_mape = self.loss_op(pred, batch.config_runtime)

                # We store #ub_size duplicates of diff_triu_vector
                # because of the technicalities of batching.
                # In actuality we need only one of them for microbatch.
                diff_triu_vector_per_ub = \
                    batch.diff_triu_vector[::self.microbatch_size]

                # loss_diff_mat is 1.0 tops
                loss_diff_mat = (1/ranking_margin) * F.margin_ranking_loss(
                    pred_diff_mat,
                    torch.zeros_like(pred_diff_mat),
                    torch.sign(diff_triu_vector_per_ub),
                    margin=ranking_margin)
                
                diff_mat_loss_scale = 1.0
                loss_diff_mat_sc = diff_mat_loss_scale * loss_diff_mat
                loss = loss_mape + loss_diff_mat_sc

                loss_val = loss.item()
                loss_mape_val = loss_mape.item()
                loss_diff_mat_sc_val = loss_diff_mat_sc.item()
                print(f"Train loss = {loss_val:.5f}, "
                      f"loss_mape = {loss_mape_val:.5f}, "
                      f"loss_diff_mat_sc = {loss_diff_mat_sc_val:.5f}, ")
                self.logger.add_scalar("train/loss", loss_val, self.iteration)
                self.logger.add_scalar("train/loss_mape", loss_mape, self.iteration)
                self.logger.add_scalar("train/loss_diff_mat_sc", loss_diff_mat_sc, self.iteration)

                kendall, p_value = kendall_rank_corrcoef(
                    pred, batch.config_runtime,
                    t_test=True, alternative='two-sided')
                print("kendall=", kendall.item(), "p_value=", p_value.item())
                self.logger.add_scalar("train/kendall", kendall.item(), self.iteration)
                self.logger.add_scalar("train/p_value", p_value.item(), self.iteration)
                self.logger.add_scalar("train/epoch", epoch, self.iteration)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                forward_backward_ts = time.time()
                forward_backward_dur = forward_backward_ts - data_load_ts

                print(f"data_load_dur={data_load_dur}, forward_backward_dur={forward_backward_dur}")
                self.logger.add_scalar("train/data_load_dur", data_load_dur, self.iteration)
                self.logger.add_scalar("train/forward_backward_dur", forward_backward_dur, self.iteration)

                iters_per_val = 10_000 if self.is_tile else 400
                if self.iteration % iters_per_val == iters_per_val - 1:
                    self.validate()
                    validation_ts = time.time()
                    validation_dur = validation_ts - forward_backward_ts
                    print(f"validation_dur={validation_dur}")
                    self.logger.add_scalar("val/validation_dur", validation_dur, self.iteration)

                end_iter_ts = time.time()

                self.iteration += 1
                if self.iteration >= max_iterations:
                    exit_training = True
                    break
            
            torch.save(self.model.state_dict(),
                       os.path.join(self.artefact_dir, "latest.pth"))
            
            epoch += (self.oversample_factor
                      if self.oversample_factor is not None else 1)
        
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
            self._compute_metrics(split, loss_list, prediction_dict)

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
        for name, dol in prediction_dict.items():
            pred_all = np.array(dol['pred_list'], dtype=np.float32)
            target_all = np.array(dol['target_list'], dtype=np.float32)

            kendall, p_value = kendall_rank_corrcoef(
                torch.tensor(pred_all), torch.tensor(target_all),
                t_test=True, alternative='two-sided')

            kendall_grand_list.append(kendall.item())
            p_value_grand_list.append(p_value.item())

        kendall_grand = np.mean(kendall_grand_list)
        p_value_grand = np.mean(p_value_grand_list)

        print(f"{split} kendall=", kendall_grand.item(), "p_value=", p_value_grand.item())
        if self.logger is not None:
            self.logger.add_scalar("val/loss", loss_grand, self.iteration)
            self.logger.add_scalar("val/kendall", kendall_grand.item(), self.iteration)
            self.logger.add_scalar("val/p_value", p_value_grand.item(), self.iteration)

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
    args = parser.parse_args()

    trainer = Trainer(args.source_data_path, args.collection, args.tag, args.debug)
    if args.test_snapshot is not None:
        trainer.load_snapshot(args.test_snapshot)
        trainer.test("test_submission.csv")
    elif args.test_val_submission_csv is not None:
        trainer.test_val_submission_csv(args.test_val_submission_csv)
    else:
        if args.start_from_pth is not None:
            trainer.load_snapshot(args.start_from_pth)
            print(f"Loading snapshot from {args.start_from_pth}")
        trainer.train()
    print("Done")


if __name__ == "__main__":
    main()
