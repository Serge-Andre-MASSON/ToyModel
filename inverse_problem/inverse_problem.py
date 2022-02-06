import torch
from torch import nn
from torch.optim.adam import Adam

import numpy as np


class InverseProblem(nn.Module):
    def __init__(self, sample_size):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(sample_size, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 4)
        )

    def forward(self, xb):
        return self.lin(xb.float())


def get_optimized_model(sample_size, learning_rate=0.05):
    model = InverseProblem(sample_size)
    opt = Adam(model.parameters(), lr=learning_rate)

    return model, opt


def loss_batch(model: InverseProblem, loss_func, xb, yb, opt: Adam = None):
    loss = loss_func(model(xb.float()), yb).float()

    if opt:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def loss_batch_for_each_parameter(model: InverseProblem, loss_func, xb, yb, opt: Adam = None):
    yb_from_model = model(xb.float())
    # TODO : refactor this
    alpha_loss = loss_func(yb_from_model[0], yb[0]).float()
    beta_loss = loss_func(yb_from_model[1], yb[1]).float()
    a_loss = loss_func(yb_from_model[2], yb[2]).float()
    b_loss = loss_func(yb_from_model[3], yb[3]).float()

    return alpha_loss, beta_loss, a_loss, b_loss


def fit(model: InverseProblem, train_dl, valid_dl, epochs, opt=Adam, loss_func=nn.MSELoss()):
    # State of things before fitting
    with torch.no_grad():
        losses, nums = zip(
            *[loss_batch(model, loss_func, xb.float(), yb) for xb, yb in valid_dl]
        )

    val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)  # ???
    print(
        f"Average validation loss before training: {val_loss}"
    )

    loss_array = [val_loss]
    alpha_loss_array = []
    beta_loss_array = []
    a_loss_array = []
    b_loss_array = []
    for _ in range(epochs):

        model.train()  # Training mode
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb.float(), yb.float(), opt)

        model.eval()  # Validation mode
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb.float(), yb) for xb, yb in valid_dl]
            )
            alpha_loss, beta_loss, a_loss, b_loss = zip(
                *[loss_batch_for_each_parameter(model, loss_func, xb.float(), yb) for xb, yb in valid_dl]
            )

        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)  # ???
        alpha_loss = np.sum(alpha_loss) / len(alpha_loss)
        beta_loss = np.sum(alpha_loss) / len(beta_loss)
        a_loss = np.sum(a_loss) / len(a_loss)
        b_loss = np.sum(b_loss) / len(b_loss)
        alpha_loss_array.append(alpha_loss)
        beta_loss_array.append(beta_loss)
        a_loss_array.append(a_loss)
        b_loss_array.append(b_loss)

        loss_array.append(val_loss)
    print(
        f"Average validation loss after training: {val_loss}"
    )
    return np.array(loss_array), np.array(alpha_loss_array), np.array(beta_loss_array), np.array(a_loss_array), np.array(b_loss_array)
