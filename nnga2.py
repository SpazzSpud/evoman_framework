import os
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller

n_hidden_neurons = 10

# Setup environments for group 1 and group 2
experiment_name_group1 = 'group1_demo_ga'
experiment_name_group2 = 'group2_demo_ga'

if not os.path.exists(experiment_name_group1):
    os.makedirs(experiment_name_group1)

if not os.path.exists(experiment_name_group2):
    os.makedirs(experiment_name_group2)

# Environment 1: enemies 7, 8
env_group1 = Environment(experiment_name=experiment_name_group1,
                         enemies=[7, 8],
                         multiplemode="yes",
                         playermode="ai",
                         player_controller=player_controller(n_hidden_neurons),
                         enemymode="static",
                         level=2,
                         speed="fastest",
                         visuals=False)

# Environment 2: enemies 2, 6
env_group2 = Environment(experiment_name=experiment_name_group2,
                         enemies=[2, 6],
                         multiplemode="yes",
                         playermode="ai",
                         player_controller=player_controller(n_hidden_neurons),
                         enemymode="static",
                         level=2,
                         speed="fastest",
                         visuals=False)

# Calculate number of weights
n_input = env_group1.get_num_sensors()
n_output = 5
n_vars = (n_input + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * n_output

# Create the folder for storing results
results_folder = 'results_ga'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)


def genetic_algorithm(pop_size, generations, mutation_rate, weight_length, env1, env2, run_id):
    population = np.random.uniform(-1, 1, (pop_size, weight_length))
    mean_fitness_over_gens_env1 = []
    max_fitness_over_gens_env1 = []
    mean_fitness_over_gens_env2 = []
    max_fitness_over_gens_env2 = []

    for gen in range(generations):
        fitness_env1 = np.array([evaluate_individual_env(ind, env1) for ind in population])
        fitness_env2 = np.array([evaluate_individual_env(ind, env2) for ind in population])

        # Calculate mean and max fitness for this generation for Env 1
        mean_fitness_env1 = np.mean(fitness_env1)
        max_fitness_env1 = np.max(fitness_env1)
        mean_fitness_over_gens_env1.append(mean_fitness_env1)
        max_fitness_over_gens_env1.append(max_fitness_env1)

        # Calculate mean and max fitness for this generation for Env 2
        mean_fitness_env2 = np.mean(fitness_env2)
        max_fitness_env2 = np.max(fitness_env2)
        mean_fitness_over_gens_env2.append(mean_fitness_env2)
        max_fitness_over_gens_env2.append(max_fitness_env2)

        # Create new population using tournament selection, crossover, and mutation
        new_population = []
        for _ in range(pop_size // 2):
            parent1 = tournament_selection(population, fitness_env1 + fitness_env2)
            parent2 = tournament_selection(population, fitness_env1 + fitness_env2)

            child1, child2 = uniform_crossover(parent1, parent2, weight_length)
            new_population.append(child1)
            new_population.append(child2)

        population = mutate(np.array(new_population), mutation_rate, weight_length).reshape(pop_size, weight_length)

        print(f"Run {run_id} - Generation {gen+1}/{generations} completed.")

    # Save the mean and max fitness for Env 1 and Env 2 separately
    np.savetxt(f'{results_folder}/ga_mean_fitness_env1_run{run_id}.txt', mean_fitness_over_gens_env1)
    np.savetxt(f'{results_folder}/ga_max_fitness_env1_run{run_id}.txt', max_fitness_over_gens_env1)
    np.savetxt(f'{results_folder}/ga_mean_fitness_env2_run{run_id}.txt', mean_fitness_over_gens_env2)
    np.savetxt(f'{results_folder}/ga_max_fitness_env2_run{run_id}.txt', max_fitness_over_gens_env2)

    return population[np.argmax(fitness_env1 + fitness_env2)]  # Return the best solution


def evaluate_individual_env(weights, env):
    # Evaluate the individual on a single environment
    fitness = env.play(pcont=weights)[0] if env.play(pcont=weights) else 0
    return fitness


def tournament_selection(population, fitness_scores, tournament_size=3):
    # Select individuals for tournament
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    best_individual_index = tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]
    return population[best_individual_index]


def uniform_crossover(parent1, parent2, weight_length):
    # Perform uniform crossover
    child1 = np.copy(parent1)
    child2 = np.copy(parent2)
    for i in range(weight_length):
        if np.random.rand() > 0.5:
            child1[i], child2[i] = parent2[i], parent1[i]
    return np.array(child1), np.array(child2)


def mutate(offspring, mutation_rate, weight_length):
    # Mutate the offspring
    offspring = offspring.reshape(-1, weight_length)
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            mutation_points = np.random.randint(weight_length, size=3)
            for point in mutation_points:
                offspring[i][point] += np.random.randn() * 0.1
    return offspring


# Run 10 independent runs
population_size = 50
generations = 100
mutation_rate = 0.1

for run_id in range(1, 11):  # Perform 10 independent runs
    best_weights = genetic_algorithm(population_size, generations, mutation_rate, n_vars, env_group1, env_group2, run_id)

print("GA training complete. Results saved in 'results_ga' folder.")
