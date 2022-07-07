import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import *


class Triple(object):
    def __init__(
        self, U_train, X_train, S_train, U_val, X_val, S_val,
    ):
        self.check_shape(U_train, X_train, S_train)
        self.check_shape(U_val, X_val, S_val)
        assert U_train.shape[1] == U_val.shape[1]
        assert X_train.shape[1] == X_val.shape[1]

        self.U_train = torch.from_numpy(U_train.astype(np.float32))
        self.X_train = torch.from_numpy(X_train.astype(np.float32))
        self.S_train = torch.from_numpy(S_train.astype(np.float32))
        self.U_val = torch.from_numpy(U_val.astype(np.float32))
        self.X_val = torch.from_numpy(X_val.astype(np.float32))
        self.S_val = torch.from_numpy(S_val.astype(np.float32))

    def print_shape(self):
        print("\n========================= Data =========================")
        print("Train: Branch input shape (#u_train, m): " + str(data.U_train.shape))
        print("       Trunk input shape  (s, dim_x):    " + str(self.X_train.shape))
        print("       Output shape       (#u_train, s): " + str(self.S_train.shape))
        print("Test:  Branch input shape (#u_test, m):  " + str(self.S_val.shape))
        print("       Trunk input shape: (s, dim_x):    " + str(self.X_val.shape))
        print("       Output shape:      (#u_test, s):  " + str(self.S_val.shape))
        print("========================================================\n")

    def check_shape(self, U, X, S):
        assert U.shape[0] == S.shape[0]
        assert X.shape[0] == S.shape[1]

    @property
    def num_funcs_train(self):
        return self.U_train.shape[0]

    @property
    def num_funcs_val(self):
        return self.U_val.shape[0]

    @property
    def num_sensors(self):
        return self.U_train.shape[1]


class Train(object):
    def __init__(self, model_path=None, device="cpu"):
        self.model_path = model_path
        self.device = device
        self.train_log = []
        self.trainloss_best = {"epoch": 0, "loss": 1e5}
        self.valloss_best = {"epoch": 0, "loss": 1e5}

    def visualize_loss(self, save=False):
        epoch = np.array([d["epoch"] for d in self.train_log if "epoch" in d])
        train_loss_log = np.array(
            [d["train_loss"] for d in self.train_log if "train_loss" in d]
        )
        val_loss_log = np.array(
            [d["val_loss"] for d in self.train_log if "val_loss" in d]
        )
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
        axes.plot(epoch, train_loss_log, label="train loss")
        axes.plot(epoch, val_loss_log, label="test loss")
        axes.legend()
        axes.set_xlabel("epochs")
        axes.set_yscale("log")
        axes.tick_params(labelsize=8)


class Train_Adam(Train):
    def __init__(
        self, batch_size, learning_rate=1e-3, model_path=None, device="cpu",
    ):
        super(Train_Adam, self).__init__(model_path, device)
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train_one_batch_adam(self, model, U_train_batch, X_train, S_train_batch):
        self.optimizer.zero_grad()
        U_train_batch, X_train, S_train_batch = (
            U_train_batch.to(self.device),
            X_train.to(self.device),
            S_train_batch.to(self.device),
        )
        preds = model(U_train_batch, X_train)
        loss = model.loss_fun(S_train_batch, preds)
        loss.backward()
        self.optimizer.step()
    

    def train_adam(self, model, data, num_epochs):
        # Early stopping on validation MSE
        patience = 20
        wait = 0
        best = 1e5
        exit_flag = False
        for epoch in range(1, num_epochs + 1):
            model.train()
            perm = np.random.permutation(data.num_funcs_train)
            for it in range(0, data.num_funcs_train, self.batch_size):
                if it + self.batch_size < data.num_funcs_train:
                    idx = perm[np.arange(it, it + self.batch_size)]
                else:
                    idx = perm[np.arange(it, data.num_funcs_train)]
                self.train_one_batch_adam(
                    model, data.U_train[idx, :], data.X_train, data.S_train[idx, :]
                )
            if epoch == 1 or epoch % 1000 == 0:
                print("--------------------------------------------------------------")
                print("Epoch %d:" % (epoch))
                preds_train = model(data.U_train[idx, :], data.X_train)
                train_loss = model.loss_fun(data.S_train[idx, :], preds_train).detach().numpy()
                preds_val = model(data.U_val, data.X_val)
                val_loss = model.loss_fun(data.S_val, preds_val).detach().numpy()

                print("train loss: %.3e, test MSE: %.3e" % (train_loss, val_loss))
                self.train_log.append(
                    {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
                )
                if train_loss < self.trainloss_best["loss"]:
                    self.trainloss_best["epoch"] = epoch
                    self.trainloss_best["loss"] = train_loss
                if val_loss < self.valloss_best["loss"]:
                    self.valloss_best["epoch"] = epoch
                    self.valloss_best["loss"] = val_loss
                if self.model_path is not None:
                    pass
                wait += 1
                if val_loss < best:
                    best = val_loss
                    wait = 0
                if wait >= patience:
                    print("Epoch %d: ... Early Stopping ..." % (epoch))
                    exit_flag = True
                    break
            if exit_flag:
                break