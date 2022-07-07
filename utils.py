import numpy as np
import matplotlib.pyplot as plt

class Normalizer:
    def __init__(self, data, eps=1e-8):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.eps = eps

    def encode(self, data):
        data = (data - self.mean) / (self.std + self.eps)
        return data

    def decode(self, data):
        data = data * (self.std + self.eps) + self.mean
        return data


def get_errors(model, y_true, y_pred):
    return (
        model.MSE(y_true, y_pred),
        model.relative_L2_Error(y_true, y_pred),
        model.Mean_Linfty_Error(y_true, y_pred),
        model.Max_Linfty_Error(y_true, y_pred),
    )

def make_triple(ufname, xfname, Sfname):
    branch_data = np.loadtxt(ufname)
    x_grid = np.loadtxt(xfname)
    trunk_data = x_grid.reshape(-1, 1)
    output_data = np.loadtxt(Sfname)
    return branch_data, trunk_data, output_data

def test(model, data):
    model.eval()
    preds = model(data.U_val, data.X_val)
    # errors of function values
    test_mse, test_L2Error, test_mean_LinfError, test_max_LinfError = get_errors(
        model, data.S_val, preds
    )
    return test_mse, test_L2Error, test_mean_LinfError, test_max_LinfError


def Plot(model, x_branch, x_trunk, output, fprefix):
    preds = model.forward(x_branch, x_trunk)
    preds = preds.detach().numpy()
    output = output.detach().numpy()
    fig1, axes1 = plt.subplots(
        len(x_branch), 2, squeeze=False, figsize=(12, 4 * len(x_branch))
    )
    for i in range(len(x_branch)):
        axes1[i][0].plot(x_trunk, preds[i], label="DeepONet")
        axes1[i][0].plot(x_trunk, output[i], label="reference")
        axes1[i][0].legend()
        axes1[i][0].set_xlabel("$x$")
        axes1[i][0].set_ylabel("$u(x,T=1)$")
        axes1[i][0].tick_params(labelsize=7)
        axes1[i][1].plot(
            x_trunk,
            preds[i] - output[i],
            label="error",
        )
        axes1[i][1].legend()
        axes1[i][1].set_xlabel("$x$")
        axes1[i][1].set_ylabel("error")
        axes1[i][1].tick_params(labelsize=7)
    fig1.savefig(fprefix + "Plots.png",bbox_inches="tight")
    # plt.show()

