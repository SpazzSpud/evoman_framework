import os
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller

experiment_name = 'nn_ga_experiment'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

hidden_neurons = 10
population_size = 50
generations = 100
mutation_rate = 0.2
crossover_rate = 0.7
enemies = [2, 5, 8]

env = Environment(
    experiment_name=experiment_name,
    enemies=enemies,
    playermode="ai",
    player_controller=player_controller(hidden_neurons),
    enemymode="static",
    level=2,
    speed="fastest",
    visuals=False,
    multiplemode="yes"
)

n_vars = (env.get_num_sensors() + 1) * hidden_neurons + (hidden_neurons + 1) * 5

def genetic_algorithm(pop_size, generations, mutation_rate, weight_length):
    population = np.random.uniform(-1, 1, (pop_size, weight_length))
    
    for gen in range(generations):
        fitness_scores = np.array([evaluate_individual(ind) for ind in population])
        offspring = crossover_and_mutate(population, fitness_scores)
        offspring_fitnesses = np.array([evaluate_individual(ind) for ind in offspring])
        population = select_new_population(population, fitness_scores, offspring, offspring_fitnesses)

    return population[np.argmax(fitness_scores)]

def evaluate_individual(weights):
    result = env.play(pcont=weights)
    if result is None:
        print("env.play() returned None for weights of length:", len(weights))
        return 0
    fitness, _, _, _ = result
    return fitness

def crossover_and_mutate(population, fitness_scores, crossover_rate=0.7, mutation_rate=0.1):
    parents = select_parents(population, fitness_scores)
    offspring = crossover(parents, crossover_rate)
    offspring = mutate(offspring, mutation_rate)
    return offspring

def select_parents(population, fitness_scores):
    selected_parents = []
    for _ in range(len(population)):
        tournament = np.random.choice(len(population), 3, replace=False)
        best_individual = tournament[np.argmax([fitness_scores[i] for i in tournament])]
        selected_parents.append(population[best_individual])
    return selected_parents

def crossover(parents, crossover_rate):
    offspring = []
    for i in range(0, len(parents), 2):
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(len(parents[i]))
            child1 = np.concatenate((parents[i][:crossover_point], parents[i+1][crossover_point:]))
            child2 = np.concatenate((parents[i+1][:crossover_point], parents[i][crossover_point:]))
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(parents[i])
            offspring.append(parents[i+1])
    return offspring

def mutate(offspring, mutation_rate):
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            mutation_point = np.random.randint(len(offspring[i]))
            offspring[i][mutation_point] += np.random.randn() * 0.1
    return offspring

if __name__ == "__main__":
    best_weights = genetic_algorithm(population_size, generations, mutation_rate, n_vars)
    print("Best solution weights:", best_weights)
