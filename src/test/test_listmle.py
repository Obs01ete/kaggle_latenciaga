import torch
import torch.nn.functional
from src.allrank_losses.listMLE import listMLE


def test_listmle():
    y_pred = torch.tensor([[1.4, 1.5, 2.5, 1.5, 1.3]])

    y_true_free = torch.tensor([[1.5, 1.33, 2.4, 7.5, 2.3]])
    loss = listMLE(y_pred, y_true_free)
    print(loss.item())

    y_true_indices = torch.argsort(y_true_free, dim=-1)
    print(y_true_indices)
    loss = listMLE(y_pred, y_true_indices.float())
    print(loss.item())

    print("Done")


def test_listmle_real():
    y_pred = torch.tensor(
        [[  -91.8735,   -96.3744,   -85.1003,   -81.5761,   -84.7840,   -89.6661, -95.5680,   -75.8551,   -79.7414,   -95.9283],
         [ -471.2412,  -468.0586,  -477.7650,  -478.6582,  -485.0116,  -487.9812, -487.4730,  -490.1606,  -481.3279,  -484.3152],
         [  -69.2004,   -73.6104,   -65.3496,   -77.7950,   -64.7972,   -81.4118, -67.2482,   -64.9661,   -85.5258,   -73.5373],
         [-1335.7797, -1335.5778, -1321.5387, -1341.9442, -1326.9027, -1320.9148, -1342.2998, -1323.1401, -1321.1099, -1333.5225]])

    y_true_free = torch.tensor(
        [[0.0615, 0.0436, 0.1317, 0.1465, 0.1314, 0.0751, 0.0458, 0.1921, 0.1520, 0.0438],
         [1.8609, 1.9057, 1.2871, 1.2488, 0.6280, 0.4249, 0.4270, 0.3855, 0.9973, 0.6298],
         [0.2329, 0.2021, 0.2635, 0.1762, 0.2657, 0.1470, 0.2466, 0.2639, 0.1169, 0.2021],
         [2.5525, 2.5867, 4.8167, 2.0521, 4.4839, 4.8277, 2.0676, 4.6074, 4.8201, 2.7756]])

    loss = listMLE(y_pred, y_true_free)
    print(loss.item())

    y_true_softmax = torch.nn.functional.softmax(y_true_free)
    loss = listMLE(y_pred, y_true_softmax)
    print(loss.item())

    free_argsort = torch.argsort(y_true_free, dim=-1)
    ranks = torch.zeros_like(free_argsort)
    for i in range(free_argsort.shape[0]):
        ranks[i, free_argsort[i]] = torch.arange(free_argsort.shape[1])
    print(ranks)
    loss = listMLE(y_pred, ranks.float())
    print(loss.item())

    print("Done")



if __name__ == "__main__":
    # test_listmle()
    test_listmle_real()
