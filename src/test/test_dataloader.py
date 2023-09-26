from pathlib import Path
from torch.utils.data import DataLoader

from src.main import concat_premade_microbatches
from src.data import LayoutData


def test_dataloader():
    data_root = Path("/home/khizbud/latenciaga/data/npz_all/npz")

    microbatch_size = 8

    collection = "layout-xla-random"

    train_data = LayoutData(
        data_root,
        coll=collection,
        split="train",
        microbatch_size=microbatch_size)

    batch_size = 5

    worker_threads = 8

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=worker_threads,
        pin_memory=True,
        collate_fn=concat_premade_microbatches)

    for i in range(10):

        for i_batch, batch in enumerate(train_loader):
            print(batch)
            # if i_batch >= 2:
            #     break
    
    print("Done")


def test_dataloader_val():
        data_root = Path("/home/khizbud/latenciaga/data/npz_all/npz")

        collection = "layout-xla-random"

        val_data = LayoutData(
            data_root,
            coll=collection,
            split="valid")

        batch_size = 10

        worker_threads = 0

        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=worker_threads,
            pin_memory=True)

        for i_batch, batch in enumerate(val_loader):
            print(batch)
            # if i_batch >= 2:
            #     break
        
        print("Done")


if __name__ == "__main__":
     test_dataloader()
    #  test_dataloader_val()
