import os
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller

n_hidden_neurons = 10

# Setup environments for group 1 and group 2
experiment_name_group1 = 'group1_demo'
experiment_name_group2 = 'group2_demo'

# Ensure directories exist
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

# Load the best solutions from both GA and DE for both environments
best_weights_ga_env1 = np.loadtxt('results/best_solution_nnga_group1.txt')
best_weights_ga_env2 = np.loadtxt('results/best_solution_nnga_group2.txt')

best_weights_de_env1 = np.loadtxt('results/best_solution_nnde_group1.txt')
best_weights_de_env2 = np.loadtxt('results/best_solution_nnde_group2.txt')

# Evaluate the best GA and DE solutions in both environments
def evaluate_solution(weights, env):
    return env.play(pcont=weights)[0]

# Evaluate GA solutions on both environments
fitness_ga_env1_on_env1 = evaluate_solution(best_weights_ga_env1, env_group1)
fitness_ga_env1_on_env2 = evaluate_solution(best_weights_ga_env1, env_group2)
fitness_ga_env2_on_env1 = evaluate_solution(best_weights_ga_env2, env_group1)
fitness_ga_env2_on_env2 = evaluate_solution(best_weights_ga_env2, env_group2)

# Evaluate DE solutions on both environments
fitness_de_env1_on_env1 = evaluate_solution(best_weights_de_env1, env_group1)
fitness_de_env1_on_env2 = evaluate_solution(best_weights_de_env1, env_group2)
fitness_de_env2_on_env1 = evaluate_solution(best_weights_de_env2, env_group1)
fitness_de_env2_on_env2 = evaluate_solution(best_weights_de_env2, env_group2)

# Print the results for comparison
print("\n--- Fitness Results ---")

# GA results
print(f"GA Best Fitness on Env 1 (tested on Env 1): {fitness_ga_env1_on_env1}")
print(f"GA Best Fitness on Env 1 (tested on Env 2): {fitness_ga_env1_on_env2}")
print(f"GA Best Fitness on Env 2 (tested on Env 1): {fitness_ga_env2_on_env1}")
print(f"GA Best Fitness on Env 2 (tested on Env 2): {fitness_ga_env2_on_env2}")

# DE results
print(f"DE Best Fitness on Env 1 (tested on Env 1): {fitness_de_env1_on_env1}")
print(f"DE Best Fitness on Env 1 (tested on Env 2): {fitness_de_env1_on_env2}")
print(f"DE Best Fitness on Env 2 (tested on Env 1): {fitness_de_env2_on_env1}")
print(f"DE Best Fitness on Env 2 (tested on Env 2): {fitness_de_env2_on_env2}")

# Optional: Calculate average fitness for GA and DE across environments
avg_fitness_ga = (fitness_ga_env1_on_env1 + fitness_ga_env2_on_env2) / 2
avg_fitness_de = (fitness_de_env1_on_env1 + fitness_de_env2_on_env2) / 2

print(f"\nAverage GA Fitness: {avg_fitness_ga}")
print(f"Average DE Fitness: {avg_fitness_de}")
