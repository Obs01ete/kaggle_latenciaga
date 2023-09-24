
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
from torch_geometric.data.data import BaseData
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


def main(): 
    # data_root = Path("/kaggle/input/predict-ai-model-runtime/npz_all/npz")
    data_root = Path("/home/khizbud/latenciaga/data/npz_all/npz")

    microbatch_size = 8 # 4

    train_data = LayoutData(data_root,
                            coll="layout-xla-random",
                            split="train",
                            microbatch_size=microbatch_size)

    logger = SummaryWriter()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device=", device)
    
    model = Model()
    model.to(device)
    
    # microbatches in batch. real batch is batch_size*microbatch_size
    batch_size = 10
    # assert batch_size == 1, ("Need to check batch.node_opcode "
    #     "is valid before enabling batch > 1")
    
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8, # 0
                              pin_memory=True,
                              collate_fn=concat_premade_microbatches)
    
    print("len(train_loader)=", len(train_loader)) # 69

    loss_op = MeanAbsolutePercentageError().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    max_iterations = 10_000
    iteration = 0
    epoch = 0
    
    exit_training = False
    while True:
        if exit_training:
            break

        model.train()

        for batch in train_loader:
            print("-"*80)
            print(iteration, batch)

            batch = batch.to(device)

            pred = model(batch.node_feat,
                        batch.node_opcode,
                        batch.node_config_feat,
                        batch.node_config_ids,
                        batch.edge_index,
                        batch.batch)
            
            print("pred", pred.detach().cpu().numpy())
            print("target", batch.config_runtime.detach().cpu().numpy())

            loss = loss_op(pred, batch.config_runtime)

            loss_val = loss.detach().cpu().item()
            print(f"Train loss = {loss_val:.5f}")
            logger.add_scalar("train/loss", loss_val, iteration)

            kendall, p_value = kendall_rank_corrcoef(
                pred, batch.config_runtime,
                t_test=True, alternative='two-sided')
            print("kendall=", kendall.item(), "p_value=", p_value.item())
            logger.add_scalar("train/kendall", kendall.item(), iteration)
            logger.add_scalar("train/p_value", p_value.item(), iteration)
            logger.add_scalar("train/epoch", epoch, iteration)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration += 1
            if iteration >= max_iterations:
                exit_training = True
                break
        
        torch.save(model.state_dict(), "latest.pth")
        
        epoch += 1
    
    print("Done")


if __name__ == "__main__":
    main()
