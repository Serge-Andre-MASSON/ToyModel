from direct_problem.grid import Grid
import numpy as np


class Coordinates():
    def __init__(self) -> None:
        pass


class RandomCoordinates(Coordinates):
    def __init__(self, grid: Grid, size: int) -> None:
        # self.X, self.T = grid.random_sample(size)
        self.X, self.T = grid.random_sample(size)
        self.grid = grid
        self.size = size

    # TODO: make this work
    def normaly_distributed_coordinates(self, grid: Grid, size):
        p = self.gaussian(grid.x, 1, 2)

        x = np.random.choice(
            grid.x,
            size=size,
            replace=False,
            p=p
        )
        t = np.random.choice(
            grid.t,
            size=size,
            # replace=False
        )

        return x, t

    def choose_coordinate_from_a_spatial_domain(self, grid: Grid, size):
        start = len(grid.x) // 4
        x = np.random.choice(
            grid.x[start:-start],
            size=size,
            # replace=False,
        )

        t = np.random.choice(
            grid.t,
            size=size,
            # replace=False,
        )
        return x, t

    def gaussian(self, x, sigma, mu):
        return np.exp(-(x - mu)**2/(2 * sigma**2)) / np.sqrt(2*np.pi * sigma**2)
