###############################################################################
# EvoMan FrameWork - V1.0 2016                                                #
# Modified: Genetic Algorithm with neural network for video game playing      #
###############################################################################

import sys
import os
import time
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller

# Setup experiment name and folder
experiment_name = 'genetic_algorithm_experiments'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Parameters for the genetic algorithm and the neural network
hidden_neurons = 10
population_size = 100
generations = 30
mutation_rate = 0.05  # Mutation probability
dom_u = 1       # Upper limit for weight values
dom_l = -1      # Lower limit for weight values
enemies = [2, 5, 8]  # List of enemies
runs_per_enemy = 5

def simulation(env, individual):
    f, p, e, t = env.play(pcont=individual)
    return f

def normalize(x, fit_pop):
    if max(fit_pop) - min(fit_pop) > 0:
        return (x - min(fit_pop)) / (max(fit_pop) - min(fit_pop))
    else:
        return 0.0000000001  # Prevent division by zero

def evaluate(population):
    return np.array([simulation(env, individual) for individual in population])

def tournament_selection(population, fitnesses):
    c1, c2 = np.random.randint(0, len(population), 2)
    return population[c1] if fitnesses[c1] > fitnesses[c2] else population[c2]

def crossover_and_mutate(population, fitnesses):
    offspring = []
    while len(offspring) < population_size:
        parent1 = tournament_selection(population, fitnesses)
        parent2 = tournament_selection(population, fitnesses)
        cross_prop = np.random.uniform(0, 1)
        child = cross_prop * parent1 + (1 - cross_prop) * parent2
        for i in range(len(child)):
            if np.random.uniform(0, 1) <= mutation_rate:
                child[i] += np.random.normal(0, 1)
            child[i] = np.clip(child[i], dom_l, dom_u)
        offspring.append(child)
    return np.array(offspring)

def selection(population, fitnesses):
    normalized_fitnesses = np.array([normalize(f, fitnesses) for f in fitnesses])
    probabilities = normalized_fitnesses / normalized_fitnesses.sum()
    chosen_indices = np.random.choice(len(population), population_size, p=probabilities, replace=False)
    return population[chosen_indices], fitnesses[chosen_indices]

def initialize_population():
    return np.random.uniform(dom_l, dom_u, (population_size, n_vars))

for enemy in enemies:
    for run in range(runs_per_enemy):
        print(f"Running GA: Enemy {enemy}, Run {run + 1}")
        env = Environment(
            experiment_name=experiment_name,
            enemies=[enemy],
            playermode="ai",
            player_controller=player_controller(hidden_neurons),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False
        )

        n_vars = (env.get_num_sensors() + 1) * hidden_neurons + (hidden_neurons + 1) * 5
        population = initialize_population()
        fitnesses = evaluate(population)

        ga_avg_fitness = []  # Store average fitness per generation
        for generation in range(1, generations + 1):
            avg_fitness = np.mean(fitnesses)
            ga_avg_fitness.append(avg_fitness)
            print(f"Generation {generation}/{generations} - Best Fitness: {max(fitnesses)}, Avg Fitness: {avg_fitness}")

            offspring = crossover_and_mutate(population, fitnesses)
            offspring_fitnesses = evaluate(offspring)
            population = np.vstack((population, offspring))
            fitnesses = np.concatenate((fitnesses, offspring_fitnesses))
            population, fitnesses = selection(population, fitnesses)

        np.savetxt(f"{experiment_name}/ga_results_enemy_{enemy}_run_{run + 1}.txt", ga_avg_fitness)
