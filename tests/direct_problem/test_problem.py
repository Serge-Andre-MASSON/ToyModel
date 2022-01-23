import numpy as np

from direct_problem import ToyProblem, Grid

alpha = 0.6
beta = 0.2
a = 1
b = 2


def solution_at_t_equal_zero(x): return np.maximum(x-2, 0)


problem = ToyProblem(
    alpha=alpha,
    beta=beta,
    a=a,
    b=b,
    solution_at_t_equal_zero=solution_at_t_equal_zero
)

x_min = 0
x_max = 5
delta_x = 0.2
t_max = 1
delta_t = 0.01

grid = Grid(
    x_min=x_min,
    x_max=x_max,
    delta_x=delta_x,
    t_max=t_max,
    delta_t=delta_t,
)


def test_set_grid():
    problem.set_grid(grid)
    assert problem.grid == grid


def test_init_boundaries():
    problem.init_boundaries()
    assert (problem.initial_condition == np.array(
        [solution_at_t_equal_zero(grid.x)])).all()
    assert (problem.left_boundary_condition == np.array([
            problem.initial_condition[0] for _ in grid.t])).all()
    assert (problem.right_boundary_condition == np.array([
            problem.initial_condition[-1] for _ in grid.t])).all()


def test_re_init():
    problem.re_init(1, 2, 3, 4)
    assert problem.alpha == 1
    assert problem.beta == 2
    assert problem.a == 3
    assert problem.b == 4
    assert problem.grid
