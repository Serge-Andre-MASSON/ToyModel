import numpy as np


class Grid():
    def __init__(self, x_min, x_max, delta_x, t_max, delta_t, t_min=0) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.delta_x = delta_x
        self.x = np.arange(x_min, x_max, delta_x)

        self.t_max = t_max
        self.delta_t = delta_t
        self.t_min = t_min
        self.t = np.arange(t_min, t_max, delta_t)

        self.coordinates = np.array(
            [[x, t] for x in self.x for t in self.t]
        )
        self.sample = None

    @property
    def shape(self):
        return len(self.x), len(self.t)

    @property
    def size(self):
        return len(self.x) * len(self.t)

    def meshed_grid(self):
        T, X = np.meshgrid(self.t, self.x)
        return X, T

    def random_sample(self, size: int):
        coordinates_len = len(self.coordinates)

        sample_indexes = np.random.choice(
            coordinates_len,
            size=size,
            replace=False
        )

        x = np.array(
            [self.coordinates[i][0] for i in sample_indexes]
        )
        t = np.array(
            [self.coordinates[i][1] for i in sample_indexes]
        )
        self.sample = x, t

        return x, t

    def index_of_x(self, x):
        return int(np.where(self.x == x)[0])

    def index_of_t(self, t):
        return int(np.where(self.t == t)[0])
