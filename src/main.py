
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader

from torchmetrics.functional.regression import kendall_rank_corrcoef

from src.data import LayoutData
from src.model import Model


def concat_premade_microbatches(microbatch_list: Sequence[Batch]):
    grand_list = []
    for micribatch in microbatch_list:
        for sample in micribatch.to_data_list():
            grand_list.append(sample)
    batch = Batch.from_data_list(grand_list)
    return batch


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # kendall, p_value = kendall_rank_corrcoef(
    #     pred, target, t_test=True, alternative='two-sided')
    # return kendall
    return torch.nn.functional.mse_loss(pred, target)


def main(): 
    # data_root = Path("/kaggle/input/predict-ai-model-runtime/npz_all/npz")
    data_root = Path("/home/khizbud/latenciaga/data/npz_all/npz")

    train_data = LayoutData(data_root,
                            coll="layout-xla-default",
                            split="train",
                            microbatch_size=4)

    # for i_microbatch, microbatch in enumerate(train_data):
    #     microbatch: Batch
    #     print(i_microbatch, microbatch)
    #     if i_microbatch >= 2:
    #         break

    logger = SummaryWriter()

    device = "gpu" if torch.cuda.is_available() else "cpu"
    
    model = Model()
    model.to(device)
    
    # microbatches in batch. real batch is batch_size*microbatch_size
    batch_size = 1 # 10
    assert batch_size == 1, ("Need to check batch.node_opcode "
        "is valid before enabling batch > 1")
    
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=concat_premade_microbatches)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for i_batch, batch in enumerate(train_loader):
        print("-"*80)
        print(i_batch, batch)

        batch = batch.to(device)

        pred = model(batch.node_feat,
                     batch.node_opcode,
                     batch.node_config_feat,
                     batch.node_config_ids,
                     batch.edge_index,
                     batch.batch)

        loss = loss_fn(pred, batch.config_runtime)

        loss_val = loss.detach().cpu().item()
        print(f"Train loss = {loss_val:.5f}")
        logger.add_scalar("train/loss", loss_val, i_batch)

        kendall, p_value = kendall_rank_corrcoef(
            pred, batch.config_runtime,
            t_test=True, alternative='two-sided')
        print("kendall=", kendall.item(), "p_value=", p_value.item())
        logger.add_scalar("train/kendall", kendall.item(), i_batch)
        logger.add_scalar("train/p_value", p_value.item(), i_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if i_batch >= 20:
        #     break
    
    print("Done")


if __name__ == "__main__":
    main()
