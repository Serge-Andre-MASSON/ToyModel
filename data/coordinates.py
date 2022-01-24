from direct_problem.grid import Grid
import numpy as np


class Coordinates():
    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.coordinates = grid.coordinates
        self.X = None
        self.T = None

    def get_random_coordinates(self, sample_size: int):
        coordinates_len = len(self.grid.coordinates)

        sample_indexes = np.random.choice(
            coordinates_len,
            size=sample_size,
            replace=False
        )

        x = np.array(
            [self.coordinates[i][0] for i in sample_indexes]
        )
        t = np.array(
            [self.coordinates[i][1] for i in sample_indexes]
        )
        self.X, self.T = x, t

        return x, t


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
