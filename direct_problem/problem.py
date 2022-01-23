import numpy as np

from direct_problem import Grid


class ToyProblem():
    def __init__(self, alpha, beta, a, b, solution_at_t_equal_zero) -> None:
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b
        self.solution_at_t_equal_zero = solution_at_t_equal_zero
        self.grid = None

    def set_grid(self, grid: Grid):
        self.grid = grid

    def init_boundaries(self):
        if not self.grid:
            raise AttributeError(
                "No grid has been defined yet : use set_grid to do so."
            )

        grid = self.grid

        self.initial_condition = self.solution_at_t_equal_zero(grid.x)
        self.left_boundary_condition = np.array([
            self.initial_condition[0] for _ in grid.t])
        self.right_boundary_condition = np.array([
            self.initial_condition[-1] for _ in grid.t])

    def re_init(self, alpha, beta, a, b,):
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b
