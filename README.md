# Google - Fast or Slow? Predict AI Model Runtime

https://www.kaggle.com/competitions/predict-ai-model-runtime

Reliable:
```
conda create -n latenciaga python=3.9
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg
pip install torchmetrics
pip install tensorboard
pip install six
```

Experimental:
```bash
conda create -n latenciaga python=3.9
conda install pyg -c pyg
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torchmetrics
pip install tensorboard
pip install six
```
