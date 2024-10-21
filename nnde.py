#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                                 		  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        general solution for enemies (games)                                         #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import sys,os

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np

experiment_name = 'controller_generalist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


n_hidden_neurons = 10
experiment_name_group1 = 'group1_demo'
if not os.path.exists(experiment_name_group1):
    os.makedirs(experiment_name_group1)

env_group1 = Environment(experiment_name=experiment_name_group1,
                         enemies=[7, 8],
                         multiplemode="yes",
                         playermode="ai",
                         player_controller=player_controller(n_hidden_neurons),
                         enemymode="static",
                         level=2,
                         speed="fastest",
                         visuals=True)

experiment_name_group2 = 'group2_demo'
if not os.path.exists(experiment_name_group2):
    os.makedirs(experiment_name_group2)

env_group2 = Environment(experiment_name=experiment_name_group2,
                         enemies=[2, 6],
                         multiplemode="yes",
                         playermode="ai",
                         player_controller=player_controller(n_hidden_neurons),
                         enemymode="static",
                         level=2,
                         speed="fastest",
                         visuals=True)

# Number of weights in the neural network (inputs * hidden neurons + hidden neurons * outputs)
n_vars = (env_group1.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

# Differential Evolution Parameters
pop_size = 100
gens = 30
F = 0.6  # Mutation factor
CR = 0.7  # Crossover probability
dom_u = 1
dom_l = -1

def initialize_population():
    return np.random.uniform(dom_l, dom_u, (pop_size, n_vars))

def fitness(env, individual):
    return env.play(pcont=individual)[0]  # Return fitness only

def mutation(pop, best, F):
    indices = np.random.choice(range(pop_size), 3, replace=False)
    x1, x2, x3 = pop[indices]
    mutant = np.clip(x1 + F * (x2 - x3), dom_l, dom_u)
    return mutant

def crossover(mutant, target, CR):
    trial = np.copy(target)
    for i in range(len(target)):
        if np.random.rand() < CR:
            trial[i] = mutant[i]
    return trial

# DE
def differential_evolution(env):
    pop = initialize_population()
    best_sol = None
    best_fitness = -np.inf

    for gen in range(gens):
        new_pop = []
        for i in range(pop_size):
            # Mutation & Crossover
            mutant = mutation(pop, best_sol, F)
            trial = crossover(mutant, pop[i], CR)
            
            # Fitness
            trial_fitness = fitness(env, trial)
            target_fitness = fitness(env, pop[i])

            # Selection
            if trial_fitness > target_fitness:
                new_pop.append(trial)
                if trial_fitness > best_fitness:
                    best_fitness = trial_fitness
                    best_sol = trial
            else:
                new_pop.append(pop[i])

        pop = np.array(new_pop)

        print(f"Generation {gen} | Best Fitness: {best_fitness}")

    return best_sol, best_fitness

print("Evolving for Group 1 (Enemies 7 and 8)")
best_sol_group1, best_fitness_group1 = differential_evolution(env_group1)
print("Evolving for Group 2 (Enemies 2 and 6)")
best_sol_group2, best_fitness_group2 = differential_evolution(env_group2)
np.savetxt(experiment_name_group1 + '/best_solution_group1.txt', best_sol_group1)
np.savetxt(experiment_name_group2 + '/best_solution_group2.txt', best_sol_group2)
print("Best solution for Group 1 saved as 'best_solution_group1.txt'.")
print("Best solution for Group 2 saved as 'best_solution_group2.txt'.")