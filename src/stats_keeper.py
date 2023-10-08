import json
import numpy as np
from attrs import asdict, define


@define
class StatsKeeper:
    iteration: int = 0
    train_kendall: float = -np.inf
    train_loss: float = np.inf
    train_mape: float = np.inf
    train_loss_diff_mat_sc: float = np.inf
    train_nz_diff_loss_frac: float = np.inf
    val_kendall: float = -np.inf
    val_loss: float = np.inf
    best_val_kendall: float = -np.inf

    def update_train(self,
                     iteration: int,
                     train_kendall: float,
                     train_loss: float,
                     train_mape: float,
                     train_loss_diff_mat_sc: float,
                     train_nz_diff_loss_frac: float,
                     ):
        self.iteration = iteration
        self.train_kendall = train_kendall
        self.train_loss = train_loss
        self.train_mape = train_mape
        self.train_loss_diff_mat_sc = train_loss_diff_mat_sc
        self.train_nz_diff_loss_frac = train_nz_diff_loss_frac

    def update_val(self,
                   iteration: int,
                   val_kendall: float,
                   val_loss: float,
                   ):
        self.iteration = iteration
        self.val_kendall = val_kendall
        self.val_loss = val_loss
        if val_kendall > self.best_val_kendall:
            self.best_val_kendall = val_kendall

    def save_as_json(self, save_path):
        with open(save_path, "w") as file:
            json.dump(asdict(self), file, indent=4)
