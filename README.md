# Google - Fast or Slow? Predict AI Model Runtime

https://www.kaggle.com/competitions/predict-ai-model-runtime

### Run experiment in docker
```shell
docker build -t latenciaga:0.1.0 -f Dockerfile .
docker run -it -v /mnt:/mnt --name latenciaga_cont latenciaga:0.1.0 bash
PYTHONPATH=. python3 src/main.py --source-data-path=/mnt/path/to/your/data/npz_all/npz/
```

Command Line Parameters:
1. --enable-wandb - don't use it (development parameter)


Reliable:
```
conda create -n latenciaga python=3.9
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg
pip install torchmetrics
pip install tensorboard
pip install six
pip install wandb
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
```
