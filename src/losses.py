from torch import nn
import torch


class MultiElementRankLoss(nn.Module):
    """
    Loss function that compares the output of the model with the output of the model with a permutation of the elements
    copied and adapted from https://www.kaggle.com/code/ksmcg90/bertlike-gat-tile-wlayout-dataset
    """

    def __init__(
            self,
            margin: float = 0.0,
            number_permutations: int = 1,
            device: str = "cpu"
            ) -> None:
        super().__init__()
        self.loss_fn = torch.nn.MarginRankingLoss(margin=margin, reduction='none')
        self.number_permutations = number_permutations
        self.device = device

    def calculate_rank_loss(self,
                            outputs: torch.Tensor,
                            config_runtime: torch.Tensor,
                            config_mask: torch.Tensor,
                            ):
        """
        Generates a permutation of the predictions and targets and calculates the loss MarginRankingLoss against the permutation
        Args:
            outputs: Tensor of shape (bs, microbs) with the outputs of the model
            config_runtime: Tensor of shape (bs, microbs) with the runtime of the model
            config_mask: Tensor of shape (bs, microbs) with 1 in the positions of the elements
            and 0 in the positions of the padding
        Returns:
            loss: Tensor of shape (1, )
        """
        bs, num_configs = outputs.shape
        src_cfg_idxs = torch.arange(bs * num_configs).view(bs, num_configs)
        permutation = torch.randperm(num_configs)
        permuted_cfg_idxs = src_cfg_idxs[:, permutation]
        # We mask those cases where we compare the same configuration
        same_cfg_mask = torch.where(permuted_cfg_idxs != src_cfg_idxs, 1, 0).to(self.device)
        permuted_runtime = config_runtime[:, permutation]
        labels = 2 * ((config_runtime - permuted_runtime) > 0) - 1
        permuted_output = outputs[:, permutation]
        permuted_config_mask = config_mask[:, permutation]
        loss = self.loss_fn(outputs.view(-1, 1), permuted_output.view(-1, 1), labels.view(-1, 1))
        loss = loss.view(bs, num_configs) * same_cfg_mask * permuted_config_mask
        return loss

    def forward(self,
                outputs: torch.Tensor,
                config_runtime: torch.Tensor,
                mask: torch.Tensor,
                ):
        loss = 0
        for _ in range(self.number_permutations):
            loss += self.calculate_rank_loss(outputs, config_runtime, mask)
        return loss / self.number_permutations


class ListMleLoss(nn.Module):
    def __init__(
            self,
            eps: float = 1e-10,
            number_permutations: int = 1,
            device: str = "cpu"
            ) -> None:
        super().__init__()
        self.number_permutations = number_permutations
        self.eps = eps
        self.device = device

    def calculate_rank_loss(self,
                            outputs: torch.Tensor,
                            config_runtime: torch.Tensor,
                            config_mask: torch.Tensor,
                            ):

        """
        ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
        :param outputs: predictions from the model, shape [batch_size, slate_length]
        :param config_runtime: ground truth labels, shape [batch_size, slate_length]
        :param config_mask: Tensor of shape (bs, microbs) with 1 in the positions of the elements
            and 0 in the positions of the padding
        :return: loss value, a torch.Tensor
        """
        # shuffle for randomised tie resolution
        bs, num_configs = outputs.shape
        random_indices = torch.randperm(num_configs)
        y_pred_shuffled = outputs[:, random_indices]
        y_true_shuffled = config_runtime[:, random_indices]
        config_mask_shuffled = config_mask[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

        # revert mask because here we mask values with 1
        mask = 1 + (-1) * config_mask_shuffled

        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        mask_sorted_by_true = torch.gather(mask, dim=1, index=indices).type(torch.bool)
        preds_sorted_by_true[mask_sorted_by_true] = float("-inf")

        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

        observation_loss = torch.log(cumsums + self.eps) - preds_sorted_by_true_minus_max

        observation_loss[mask_sorted_by_true] = 0.0

        return observation_loss  #torch.mean(torch.sum(observation_loss, dim=1))

    def forward(self,
                outputs: torch.Tensor,
                config_runtime: torch.Tensor,
                mask: torch.Tensor,
                ):
        loss = 0
        for _ in range(self.number_permutations):
            loss += self.calculate_rank_loss(outputs, config_runtime, mask)
        return loss / self.number_permutations