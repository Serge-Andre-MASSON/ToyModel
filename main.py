from plotter.plot import compare_original_solution_with_reconstructed_solution

from inverse_problem import get_optimized_model, fit
from time import time

import numpy as np
import torch
from torch import float


from direct_problem import Grid
from data.data_generator import DataGenerator, RandomParameters, RandomCoordinates
from direct_problem.problem import ToyProblem
from direct_problem.solver import CSFTSolver

# On se donne une grille sur laquelle plusieurs
# Toy_problem devront être résolus.
t_0 = time()
GRID = Grid(
    x_min=0,
    x_max=5,
    delta_x=0.2,
    t_max=1,
    delta_t=0.01
)

grid_size = GRID.size


def SOLUTION_AT_T_EQUAL_ZERO(x): return np.maximum(2-x, 0)
# SOLUTION_AT_T_EQUAL_ZERO = np.sin


# On choisit le nombre de points dont les valeurs
# seront supposées connues
SAMPLE_SIZE = grid_size // 5


# On choisit aléatoirement SAMPLE_SIZE points sur la grille
coordinates = RandomCoordinates(GRID, SAMPLE_SIZE)
X, T = coordinates.X, coordinates.T  # OK


# On se donne le nombre de problemes que l'on va résoudre
# pour générer les données.
NUMBERS_OF_PROBLEMS = 500


# Génération des paramètres. dim=2 indique que l'on est sur un
# systeme à deux équations : 4 paramètres par probleme.
DIM = 2
parameters = RandomParameters(NUMBERS_OF_PROBLEMS, dim=DIM)


data_generator = DataGenerator(
    coordinates,
    parameters,
    SOLUTION_AT_T_EQUAL_ZERO)

BATCH_SIZE = 30

u = data_generator.generate_data()
print(u)
print(
    f"""Les données d'entrées sont sous la forme d'un tableau numpy de {u.shape[0]} lignes,
chacune représentant un problème, et de {u.shape[1]} colonnes, représentant le nombre
de valeurs connues pour chaque problème.""")
train_dl, test_dl = data_generator.dataloader(batch_size=BATCH_SIZE)
print(f"Temps total pour générer le jeu de données: {time() - t_0}\n")


model, opt = get_optimized_model(SAMPLE_SIZE)
epochs = 70
t_1 = time()

loss_array = fit(model, train_dl, test_dl, epochs=epochs, opt=opt)
print(f"Temps de l'entrainement pour {epochs} epochs: {time() - t_1}\n")


alpha = 0.4
beta = 0.2
a = 1
b = 2

real_problem = ToyProblem(
    alpha,
    beta,
    a,
    b,
    SOLUTION_AT_T_EQUAL_ZERO)

real_problem.set_grid(GRID)
real_problem.init_boundaries()

solver = CSFTSolver(real_problem)
u, v = solver.solve()


u_observed = torch.tensor(np.array([
    u[GRID.index_of_x(x), GRID.index_of_t(t)]for x, t in zip(X, T)
]))

r_parameters = model(u_observed).detach().numpy()
r_problem = ToyProblem(
    r_parameters[0],
    r_parameters[1],
    r_parameters[2],
    r_parameters[3],
    SOLUTION_AT_T_EQUAL_ZERO)

r_problem.set_grid(GRID)
r_problem.init_boundaries()

r_solver = CSFTSolver(r_problem)
u_r, v_r = r_solver.solve()

compare_original_solution_with_reconstructed_solution(
    u, v, u_r, v_r, GRID, loss_array)
