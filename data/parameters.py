import numpy as np
import pandas as pd


class Parameters():
    def __init__(self) -> None:
        pass


class RandomParameters(Parameters):
    def __init__(
        self,
        parameters_size: int,
        max_transition=3,
        dim: int = 2
    ) -> None:
        self.parameters_size = parameters_size
        self.diffusion_array = np.random.rand(parameters_size, dim)

        self.transition_array = \
            np.random.rand(parameters_size, dim) * max_transition

        self.parameters_array = np.concatenate(
            (self.diffusion_array,
             self.transition_array),
            axis=1)

    def __repr__(self) -> str:
        df = pd.DataFrame(data=self.parameters_array,
                          columns=["alpha", "beta", "a", "b"])
        return df.head().__repr__()
