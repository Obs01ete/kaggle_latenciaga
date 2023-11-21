# Solution to "Google - Fast or Slow? Predict AI Model Runtime" competition on Kaggle

Our team Latenciaga took 2nd place in the competition.

Team members:
1. Dmitrii Khizbullin (team leader)
2. Denis Divitsky
3. David Pugh
4. Ivan Isaev
5. Kirill Snezhko

The link to the competition:

https://www.kaggle.com/competitions/predict-ai-model-runtime

We express our gratitude to Kaggle as well as Google’s TPU team for organizing this remarkable challenge.

Please see [How To Run](how_to_run.md) for the instructions on how to reproduce the results.

Partially inspired by this competition, Dmitrii published an article [Ten Patterns and Antipatterns of Deep Learning Experimentation](https://pub.towardsai.net/ten-patterns-and-antipatterns-of-deep-learning-experimentation-e91bb0f6feda) at Towards AI.

## Introduction
Our implementation is a SageConv-based graph neural network (GNN) operating on whole graphs and trained in PyTorch/PyTorch-Geometric. The GNN was trained with the help of one or two losses, including a novel DiffMat loss, which we will discuss later.

## Dataset preprocessing
We preprocess data from all 5 subsets by removing duplicates by config. We discovered that for each graph, several instances of configurations (all node-wise concatenated together for Layout and subgraph for Tile) are identical, while the corresponding runtimes are different with a 0.4% max-to-min difference. We reduce these groups by a minimum. For Layout-XLA, we filtered out all Unet graphs since we identified that `unet_3d.4x4.bf16` is badly corrupted. For the same reason, we removed `mlperf_bert_batch_24_2x2` from Layout-XLA-Default validation to improve the stability of the validation. We identified many other graphs whose data is seemingly corrupted, but we did not filter them out. As a part of preprocessing, we repack the NPZs for Layout so that for each graph, each config+runtime measurement (out of 100k or less) can be loaded from NPZ individually without loading the entire NPZ. With this repacking, thanks to lazy loading, random reads were accelerated 5-10 times, resulting in a similar reduction of training wall clock time, whereas the training became GPU-bound instead of data-loading bound.

## Model
We train 5 models from scratch, one for each subset, applying different hyperparameters as summarized in the table below. All GNN layers are SageConv layers with residual connections whenever the number of input and output channels are the same.

| subsets | layers x channels | # parameters |
| -- | -- | -- |
| Layout-XLA | 2x64 + 2x128 + 2x256 | 270k |
| Layout-NLP & Tile | 4x256 + 4x512 | 2.3M |

</br>

The node types are embedded into 12 dimensions. Node features are compressed with `sign(x)*log(abs(x))` and shaped into 20 dimensions by a linear layer. For Layout, the configs are not transformed; for Tile, the graph configuration is broadcast to all nodes. We apply early fusion by combining the three above into a single feature vector before passing it to GNN layers. Features produced by the GNN layer stack are transformed to one value per node and then sum-reduced to form a single graph-wise prediction. 

## Training procedure

We follow training and validation splits provided by the competition authors. For all 5 subsets, the training was only performed on a training split.

The batch is organized into 2 levels of hierarchy: the upper level is different graphs, and the lower level is the same graph and different configurations, grouped in microbatches of the same size (also known as slates). This procedure allows applying ranking loss to the group of samples within a microbatch. We found that using some sort of ranking loss is essential for the score. Models trained with a ranking loss (ListMLE, MarginRankingLoss) heavily outperformed element-wise losses (MAPE, etc). 

| hyperparameter | Tile subset | Layout- XLA-Random | Layout- XLA-Default | Layout- NLP-Random | Layout- NLP-Default |
| --- | --- | --- | --- | --- | --- |
| microbatch size | 10 | 4 | 4 | 10 | 10 |
| number of microbatches in a batch | 100 | 10 | 10 | 4 | 4 |
| batch size | 1000 | 40 | 40 | 40 | 40 |

</br>

The following hyperparameters were set:
1. Adam/AdamW optimizer,
2. Learning rate 1e-3,
3. 400k iterations,
4. Step learning rate scheduler at 240k, 280k, 320k, and 360k by factor of `1/sqrt(10)`.

Training time is approximately 20 hours on A100 for each of 5 subsets. No early stopping was employed. All snapshots for submission were taken from the 400k-th iteration.

Losses used for training:
1. ListMLE for Layout-NLP,
2. A novel DiffMat loss for Tile,
3. For Layout-XLA, it is a combination of 2 losses: the DiffMat loss and MAPE loss.

For ListMLE loss, we used prediction norm-clipping to avoid numerical instability resulting from dividing a big number by a big number. We do not use prediction L2 normalization before ListMLE loss since we find it damages the score.

The novel DiffMat loss is described with the following algorithm. Within a microbatch, a full antisymmetric matrix of pairwise differences is constructed for the predictions and for the targets. The upper triangular matrix is taken from the difference matrix and flattened. Margin Ranking Loss with a margin of 0.01 is applied between predicted values and zeros. This novel loss, combined with MAPE loss, consistently outperformed ListMLE on XLA.

![difffmat](https://raw.githubusercontent.com/Obs01ete/latenciaga_materials/191000e669012bb9d8cae52a309d73ac4d9a57a7/assets/diffmat.png)

## Remarks on the validation (CV) stability
We found Kendall tau on the validation splits extremely unstable for XLA Random and Default since the dataset is relatively small, and there is a significant domain gap between train and validation, and presumably test. Repeats of training results in up to 13 percentage points of difference between outcomes. 

## Experiments that did not work

### Data filtration
Some graphs’ data is badly damaged. For example, `magenta_dynamic` has the following rollout of runtimes vs config ID. In no way can these be measurements from the same graph. 

![](https://raw.githubusercontent.com/Obs01ete/latenciaga_materials/main/assets/damaged1.png)

Below are other examples where we are unsure about the conditions in which these measurements were performed.

![](https://raw.githubusercontent.com/Obs01ete/latenciaga_materials/main/assets/damaged2.png)
![](https://raw.githubusercontent.com/Obs01ete/latenciaga_materials/main/assets/damaged3.png)

Nevertheless, we do not filter out these graphs and others since we could not reliably observe the improvement from their removal due to the earlier mentioned instability of validation Kendall numbers.

### Data recovery
We tried to find the damaged data and remove it in an automatic manner by computing block-wise entropy of the runtimes between adjacent blocks. While the detection seems to work visually, we observed a negative impact on the score and did not proceed with this feature.

Example 1:

![link](https://raw.githubusercontent.com/Obs01ete/latenciaga_materials/main/assets/entropy1.png)

Example 2:

![link](https://raw.githubusercontent.com/Obs01ete/latenciaga_materials/main/assets/entropy2.png)

Before and after entropy filtration:

![link](https://raw.githubusercontent.com/Obs01ete/latenciaga_materials/main/assets/entropy3.png)


### Other experiments we tried that did NOT work:
1. GATv2Conv, GATv2 backbone, GINEConv,
2. Dropout,
3. Training on merged Random and Default - hurts both,
4. Adding reverse edges,
5. Online hard negative mining (OHEM) - did not help since train loss is nowhere near zero,
6. Train blindly on the merged train and valid (trainval),
7. Train 4 folds and merge by mean latency and by mean reciprocal rank (MRR),
8. Periodic LR schedule.

## Conclusion
We found Google Fast or Slow to be a great competition, and we enjoyed it a lot, along with learning many new things, especially ranking losses.
