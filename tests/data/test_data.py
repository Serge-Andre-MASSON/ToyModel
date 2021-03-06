import numpy as np

from data import DataGenerator, RandomParameters
from direct_problem import Grid

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

sample_size = 10
parameter_size = 50
def solution_at_t_equal_zero(x): return np.maximum(x-2, 0)


parameters = RandomParameters(parameters_size=parameter_size)

data_generator = DataGenerator(grid=grid, parameters=parameters,
                               solution_at_t_equal_zero=solution_at_t_equal_zero)
