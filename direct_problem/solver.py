import numpy as np
import matplotlib.pyplot as plt

from direct_problem import ToyProblem


class CSFTSolver():
    def __init__(self, toy_problem: ToyProblem) -> None:

        self.toy_problem = toy_problem
        self.grid = toy_problem.grid

        self.u = np.zeros(self.grid.shape, dtype=float)
        self.v = np.zeros(self.grid.shape, dtype=float)

        self.u[:, 0] = self.toy_problem.initial_condition
        self.v[:, 0] = self.toy_problem.initial_condition

        self.u[0, 1::] = self.toy_problem.left_boundary_condition[1:]
        self.v[0, 1::] = self.toy_problem.left_boundary_condition[1:]

        self.u[-1, 1::] = self.toy_problem.right_boundary_condition[1:]
        self.v[-1, 1::] = self.toy_problem.right_boundary_condition[1:]

        self.set_derivative_matrix()
        self.set_coupling_terms()

        self.is_solved = False

    def set_derivative_matrix(self):
        delta_t = self.grid.delta_t
        delta_x = self.grid.delta_x

        nrows = len(self.grid.x) - 2
        ncols = len(self.grid.x)

        alpha = self.toy_problem.alpha
        a = self.toy_problem.a

        u_cfl = alpha*delta_t/(delta_x**2)
        u_cfl_array = np.array(
            [u_cfl, 1 - 2*u_cfl + a*delta_t, u_cfl]
        )

        beta = self.toy_problem.beta
        b = self.toy_problem.b

        v_cfl = beta*delta_t/(delta_x**2)
        v_cfl_array = np.array(
            [v_cfl, 1 - 2*v_cfl + b*delta_t, v_cfl]
        )

        self.u_solving_matrix = np.zeros((nrows, ncols))
        self.v_solving_matrix = np.zeros((nrows, ncols))

        for i in range(nrows):
            self.u_solving_matrix[i, i:i+3] = u_cfl_array
            self.v_solving_matrix[i, i:i+3] = v_cfl_array

    def set_coupling_terms(self):
        self.coupling_term_for_u = self.toy_problem.a * self.grid.delta_t
        self.coupling_term_for_v = self.toy_problem.b * self.grid.delta_t

    def solve(self):
        self.is_solved = True
        for j in range(1, len(self.grid.t)):
            self.u[1:-1, j] = np.dot(self.u_solving_matrix, self.u[:, j-1]) \
                - self.coupling_term_for_u * self.v[1:-1, j-1]
            self.v[1:-1, j] = np.dot(self.v_solving_matrix, self.v[:, j-1]) \
                - self.coupling_term_for_v * self.u[1:-1, j-1]

        return self.u, self.v

    def plot(self, notebook=False):
        if self.is_solved:
            u, v = self.u, self.v
        else:
            u, v = self.solve()

        X, T = self.toy_problem.grid.meshed_grid()

        fig = plt.figure(figsize=(15, 15))
        u_ax = fig.add_subplot(121, projection="3d")
        v_ax = fig.add_subplot(122, projection="3d")

        u_ax.set_title("$u$")
        v_ax.set_title("$v$")

        u_ax.plot_surface(X, T, u)
        v_ax.plot_surface(X, T, v)

        if not notebook:
            return fig
