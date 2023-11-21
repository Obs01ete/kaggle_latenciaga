### Launch training

See `train_layout_*_long.job` and `train_tile_xla.job` scripts on how to launch training. The partial submission csv is saved to the artefacts directory once training is finished.

Out of the 5 subsets, some of them were trained from different git tags. See the mapping of the tags and subsets below:
1. `good_for_nlp` for Layout-NLP-Random and Layout-NLP-Default.
2. `last_diff_mat_loss_good_for_xla` for Layout-XLA-Random, Layout-XLA-Default and Tile-XLA.

### Run experiment in docker
```shell
docker build -t latenciaga:0.1.0 -f Dockerfile .
docker run -it -v /mnt:/mnt --name latenciaga_cont latenciaga:0.1.0 bash
PYTHONPATH=. python3 src/main.py --source-data-path=/mnt/path/to/your/data/npz_all/npz/
```

Command Line Parameters:
1. --enable-wandb - don't use it (development parameter)

### Conda setup

Reliable:
```
conda create -n latenciaga python=3.9
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg
pip install torchmetrics
pip install tensorboard
pip install six
pip install wandb
pip install attrs
```

Experimental:
```bash
conda create -n latenciaga python=3.9
conda install pyg -c pyg
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torchmetrics
pip install tensorboard
pip install six
pip install wandb
pip install attrs
pip install pandas
```
