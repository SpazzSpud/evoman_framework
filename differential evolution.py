################################
# EvoMan FrameWork - V1.0 2016 #
# Modified: Differential Evolution neural network.            #
################################

import numpy as np
import os
from evoman.environment import Environment
from demo_controller import player_controller

experiment_name = 'differential_evolution_experiments'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

hidden_neurons = 10
n_vars = None  # Will be set dynamically for each enemy
npop = 100  # Population size
gens = 30   # Number of generations
F = 0.8     # Mutation factor
CR = 0.8    # Crossover probability
dom_l, dom_u = -1, 1  
enemies = [2, 5, 8]
runs_per_enemy = 5

def simulation(env, individual):
    f, p, e, t = env.play(pcont=individual)
    return f

def evaluate_population(env, pop):
    return np.array([simulation(env, individual) for individual in pop])

def mutate(pop, index, F):
    indices = [i for i in range(npop) if i != index]
    a, b, c = pop[np.random.choice(indices, 3, replace=False)]
    mutant = np.clip(a + F * (b - c), dom_l, dom_u)
    return mutant

def crossover(mutant, target, CR):
    trial = np.copy(target)
    for i in range(len(target)):
        if np.random.rand() < CR:
            trial[i] = mutant[i]
    return trial

def select(trial, target, trial_fit, target_fit):
    if trial_fit > target_fit:
        return trial, trial_fit
    else:
        return target, target_fit

for enemy in enemies:
    for run in range(runs_per_enemy):
        print(f"Running DE: Enemy {enemy}, Run {run + 1}")
        env = Environment(experiment_name=experiment_name,
                          enemies=[enemy],
                          playermode="ai",
                          player_controller=player_controller(hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=False)

        n_vars = (env.get_num_sensors() + 1) * hidden_neurons + (hidden_neurons + 1) * 5
        pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
        fit_pop = evaluate_population(env, pop)
        
        de_avg_fitness = []  # Store average fitness per generation
        for gen in range(gens):
            new_pop = np.copy(pop)
            new_fit_pop = np.copy(fit_pop)

            for i in range(npop):
                mutant = mutate(pop, i, F)
                trial = crossover(mutant, pop[i], CR)
                trial_fit = simulation(env, trial)
                new_pop[i], new_fit_pop[i] = select(trial, pop[i], trial_fit, fit_pop[i])

            pop, fit_pop = new_pop, new_fit_pop

            avg_fitness = np.mean(fit_pop)
            de_avg_fitness.append(avg_fitness)
            print(f"Generation {gen}: Best Fitness = {np.max(fit_pop)}, Avg Fitness = {avg_fitness}")

        np.savetxt(f"{experiment_name}/de_results_enemy_{enemy}_run_{run + 1}.txt", de_avg_fitness)
