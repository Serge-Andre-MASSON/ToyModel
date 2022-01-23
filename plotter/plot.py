import matplotlib.pyplot as plt

from direct_problem import ToyProblem, Grid, CSFTSolver


def plot_solutions_of_direct_toy_problem(problem: ToyProblem, grid: Grid, show=True):
    problem.set_grid(grid)
    problem.init_boundaries()

    solver = CSFTSolver(problem)

    u, v = solver.solve()
    X, T = grid.meshed_grid()

    fig = plt.figure("$u$, $v$ et $u - v$", figsize=(15, 15))
    u_ax = fig.add_subplot(121, projection="3d")
    v_ax = fig.add_subplot(122, projection="3d")

    u_ax.set_title("$u$")
    v_ax.set_title("$v$")

    u_ax.plot_surface(X, T, u)
    v_ax.plot_surface(X, T, v)

    # if show:
    #     plt.show()

    # return fig


def compare_original_solution_with_reconstructed_solution(u, v, u_r, v_r, grid, loss_array, plot=True):
    fig = plt.figure("Original and reconstruct solution", figsize=(15, 15))
    u_ax = fig.add_subplot(331, projection="3d")
    u_r_ax = fig.add_subplot(332, projection="3d")
    diff_u_ax = fig.add_subplot(333, projection="3d")
    v_ax = fig.add_subplot(334, projection="3d")
    v_r_ax = fig.add_subplot(335, projection="3d")
    diff_v_ax = fig.add_subplot(336, projection="3d")
    loss_ax = fig.add_subplot(338)

    u_ax.set_title("$u$ à reconstruire")
    u_r_ax.set_title("$u$ reconstruit")
    diff_u_ax.set_title("Erreur")

    v_ax.set_title("$v$ à reconstruire")
    v_r_ax.set_title("$v$ reconstruit")
    diff_v_ax.set_title("Erreur")

    loss_ax.set_title(
        "Evolution de l'erreur au cours de l'apprentissage du réseau")

    mesh_X, mesh_T = grid.meshed_grid()
    u_ax.plot_surface(mesh_X, mesh_T, u)
    u_r_ax.plot_surface(mesh_X, mesh_T, u_r)
    diff_u_ax.plot_surface(mesh_X, mesh_T, u - u_r)

    v_ax.plot_surface(mesh_X, mesh_T, v)
    v_r_ax.plot_surface(mesh_X, mesh_T, v_r)
    diff_v_ax.plot_surface(mesh_X, mesh_T, v - v_r)

    loss_ax.plot(loss_array)
    if plot:
        plt.show()

    return fig
