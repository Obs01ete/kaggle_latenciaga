
from pathlib import Path
from typing import Sequence, Dict, Optional
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from torchmetrics.functional.regression import kendall_rank_corrcoef
from torchmetrics import MeanAbsolutePercentageError

from src.data import LayoutData
from src.model import Model


def concat_premade_microbatches(microbatch_list: Sequence[Batch]):
    grand_list = []
    for micribatch in microbatch_list:
        for sample in micribatch.to_data_list():
            grand_list.append(sample)
    batch = Batch.from_data_list(grand_list)
    return batch


class Trainer:
    def __init__(self, debug: bool = False):
        self.debug = debug
    
        data_root = Path("/home/khizbud/latenciaga/data/npz_all/npz")

        self.microbatch_size = 4

        self.collection = "layout-xla-random"

        self.train_data = LayoutData(
            data_root,
            coll=self.collection,
            split="train",
            microbatch_size=self.microbatch_size)

        self.val_data = LayoutData(
            data_root,
            coll=self.collection,
            split="valid")

        self.test_data = LayoutData(
            data_root,
            coll=self.collection,
            split="test")
        
        default_worker_threads = 16 # 8
        self.worker_threads = 0 if self.debug else default_worker_threads

        self.logger = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device=", self.device)
        
        self.model = Model()
        self.model.to(self.device)

        # microbatches in batch. real batch is batch_size*microbatch_size
        self.batch_size = 10
        # assert self.batch_size == 1, ("Need to check batch.node_opcode "
        #     "is valid before enabling batch > 1")
        self.val_batch_size = 40
        self.test_batch_size = self.val_batch_size

        self.loss_op = MeanAbsolutePercentageError().to(self.device)

    def train(self):

        if self.logger is None:
            self.logger = SummaryWriter()

        self.iteration = 0

        # self.validate()

        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.worker_threads,
            pin_memory=True,
            collate_fn=concat_premade_microbatches)
        
        print("len(train_loader)=", len(train_loader)) # 69
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        max_iterations = 100_000 # 40_000
        epoch = 0
        
        exit_training = False
        while True:
            if exit_training:
                break

            for batch in train_loader:
                print("-"*80)
                print(self.iteration, batch)

                batch = batch.to(self.device)

                self.model.train()

                pred = self.model(
                    batch.node_feat,
                    batch.node_opcode,
                    batch.node_config_feat,
                    batch.node_config_ids,
                    batch.edge_index,
                    batch.batch)
                
                # print("pred", pred.detach().cpu().numpy())
                # print("target", batch.config_runtime.detach().cpu().numpy())

                loss = self.loss_op(pred, batch.config_runtime)

                loss_val = loss.detach().cpu().item()
                print(f"Train loss = {loss_val:.5f}")
                self.logger.add_scalar("train/loss", loss_val, self.iteration)

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

                iters_per_val = 100
                if self.iteration % iters_per_val == iters_per_val - 1:
                    self.validate()

                self.iteration += 1
                if self.iteration >= max_iterations:
                    exit_training = True
                    break
            
            torch.save(self.model.state_dict(), "latest.pth")
            
            epoch += 1
        
        self.test("end_of_train_submission.csv")
    
    def validate(self):
        self._validate('valid')
    
    def test(self, submission_csv_path: str):
        self._validate('test', submission_csv_path)
    
    def _validate(self, split: str,
                  submission_csv_path: Optional[str] = None):

        assert split in {'valid', 'test'}

        if submission_csv_path is None:
            submission_csv_path = "submission_val.csv"

        loader = DataLoader(
            self.val_data if split == 'valid' else self.test_data,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.worker_threads,
            pin_memory=True)
        
        print("len(loader)=", len(loader))

        dd = defaultdict(lambda: defaultdict(list))
        loss_list = []

        for i_batch, batch in tqdm(enumerate(loader)):

            batch = batch.to(self.device)

            self.model.eval()

            with torch.no_grad():
                pred = self.model(
                    batch.node_feat,
                    batch.node_opcode,
                    batch.node_config_feat,
                    batch.node_config_ids,
                    batch.edge_index,
                    batch.batch)

                if split == 'valid':
                    loss = self.loss_op(pred, batch.config_runtime)
                    loss_list.append(loss.item())

            # for key in batch.keys:
            #     v = getattr(batch, key)
            #     if isinstance(v, torch.Tensor):
            #         print(key, v.shape)

            for pr, tg, fn in zip(pred, batch.config_runtime, batch.fname):
                assert len(fn) == 1
                dol = dd[fn[0]]
                dol['pred_list'].append(pr.item())
                dol['target_list'].append(tg.item())

            if self.debug:
                if i_batch >= 2:
                    break
        
        if split == 'valid':
            loss_grand = np.mean(np.array(loss_list, dtype=np.float32))

            kendall_grand_list = []
            p_value_grand_list = []
            for name, dol in dd.items():
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
            print(f"{split} loss = {loss_grand:.5f}")
            if self.logger is not None:
                self.logger.add_scalar("val/loss", loss_grand.item(), self.iteration)
                self.logger.add_scalar("val/kendall", kendall_grand.item(), self.iteration)
                self.logger.add_scalar("val/p_value", p_value_grand.item(), self.iteration)

        # Prepare submission
        submission_dict: Dict[str, np.ndarray] = dict()
        for name, dol in dd.items():
            pred_all = np.array(dol['pred_list'], dtype=np.float32)
            # NB: pred_all are guaranteed to be sequential by DataLoader
            submission_ranks = np.argsort(pred_all)
            submission_dict[name] = submission_ranks

        self.save_submission(submission_dict, submission_csv_path)
        print(f"Saved to {submission_csv_path}")

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


def main():
    parser = ArgumentParser(description='Latenciaga')
    parser.add_argument('--test-snapshot', action='store', type=str,
                        help='Provide .pth, get submission.csv')

    args = parser.parse_args()

    trainer = Trainer()
    if args.test_snapshot is not None:
        trainer.load_snapshot(args.test_snapshot)
        trainer.test("test_submission.csv")
    else:
        trainer.train()
    print("Done")


if __name__ == "__main__":
    main()
