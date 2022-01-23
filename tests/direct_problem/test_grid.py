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


def test_grid_shape_and_size():
    len_x = (x_max - x_min)/delta_x
    len_t = t_max/delta_t
    assert grid.shape == (len_x, len_t)
    assert grid.size == len_x * len_t


def test_random_sample():
    size = 3
    x, t = grid.random_sample(3)
    assert len(x) == len(t) == 3


def test_index_of():
    x, t = grid.random_sample(1)
    # x, t = x[0], t[0]
    # x, t = 3 * grid.delta_x, 5 * grid.delta_t

    assert x in grid.x
    assert t in grid.t

    assert grid.x[grid.index_of_x(x)] == x
    assert grid.t[grid.index_of_t(t)] == t
