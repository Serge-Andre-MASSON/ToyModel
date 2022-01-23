import numpy as np
from numpy import ndarray, double
from torch import tensor
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader

from direct_problem.problem import ToyProblem
from direct_problem.solver import CSFTSolver

from data import RandomCoordinates
from data import RandomParameters


# TODO: Rendre plus générique le code suivant en le généralisant
# à tout les types possibles de classes induites et aux
# dimensions supérieures à deux.

class DataGenerator():
    def __init__(self, coordinates: RandomCoordinates, parameters: RandomParameters, solution_at_t_equal_zero) -> None:
        self.grid = coordinates.grid
        self.coordinates = coordinates
        self.parameters = parameters
        self.solution_at_t_equal_zero = solution_at_t_equal_zero
        self.test_parameters_size = int(self.parameters.parameters_size // 5)

    def generate_data(self):
        toy_problem = ToyProblem(0, 0, 0, 0, self.solution_at_t_equal_zero)
        toy_problem.set_grid(self.grid)
        toy_problem.init_boundaries()

        X = self.coordinates.X
        T = self.coordinates.T
        solution_values = []

        # Ce code ne s'applique que dans le cas où
        # self.parameters.dim = 2.
        for i in range(self.parameters.parameters_size):
            alpha, beta, a, b = self.parameters.parameters_array[i]
            toy_problem.re_init(alpha, beta, a, b)
            u = CSFTSolver(toy_problem).solve()[0]
            solution_values.append([u[self.grid.index_of_x(X[k]),
                                      self.grid.index_of_t(T[k])] for k in range(len(X))])

        return np.array(solution_values, dtype=np.float64)

    def split(self, array: ndarray, here: int):
        train_data = array[here:]
        test_data = array[:here]
        return train_data, test_data

    def dataset(self):
        solutions_array = self.generate_data()
        x_train, x_test = self.split(
            solutions_array, self.test_parameters_size)
        y_train, y_test = self.split(
            self.parameters.parameters_array, self.test_parameters_size)
        train_ds = TensorDataset(
            tensor(x_train),
            tensor(y_train),)
        test_ds = TensorDataset(
            tensor(x_test),
            tensor(y_test))
        return train_ds, test_ds

    def dataloader(self, batch_size):
        train_ds, test_ds = self.dataset()
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=2 * batch_size)
        return train_dl, test_dl

    def __repr__(self) -> str:
        # TODO: Fxer quelque part les données générées pour ne
        # pas avoir à les recalculer à chaque fois
        solutions_array = self.generate_data()
        # df = pd.DataFrame(data=solutions_array,
        #                   columns=[f"u at P_{i}" for i in range(len(solutions_array[0]))])
        return solutions_array.__repr__()
