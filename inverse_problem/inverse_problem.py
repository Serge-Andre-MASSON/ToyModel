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


def fit(model: InverseProblem, train_dl, valid_dl, epochs, opt=Adam, loss_func=nn.MSELoss()):
    # State of things before fitting
    with torch.no_grad():
        losses, nums = zip(
            *[loss_batch(model, loss_func, xb.float(), yb) for xb, yb in valid_dl]
        )

    val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)  # ???
    print(
        f"Average validation loss before fit: {val_loss}"
    )

    loss_array = [val_loss]
    for _ in range(epochs):

        model.train()  # Training mode
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb.float(), yb.float(), opt)

        model.eval()  # Validation mode
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb.float(), yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)  # ???

        loss_array.append(val_loss)
    print(
        f"Average validation loss: {val_loss}"
    )
    return np.array(loss_array)
