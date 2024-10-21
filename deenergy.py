import os
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller

n_hidden_neurons = 10

# Load best solutions for DE
best_solution_group1_de = np.loadtxt('results_de/best_solution_group1.txt')
best_solution_group2_de = np.loadtxt('results_de/best_solution_group2.txt')

# Ensure directories exist for logging
if not os.path.exists('de_test_group1'):
    os.makedirs('de_test_group1')

if not os.path.exists('de_test_group2'):
    os.makedirs('de_test_group2')

# Create environments for DE
env_group1_de = Environment(experiment_name='de_test_group1', enemies=[1], multiplemode="no", 
                            playermode="ai", player_controller=player_controller(n_hidden_neurons), 
                            enemymode="static", level=2, speed="fastest", visuals=False)

env_group2_de = Environment(experiment_name='de_test_group2', enemies=[1], multiplemode="no", 
                            playermode="ai", player_controller=player_controller(n_hidden_neurons), 
                            enemymode="static", level=2, speed="fastest", visuals=False)

# Function to calculate energy points for DE
def calculate_energy_points(env, best_weights):
    player_energy_list = []
    enemy_energy_list = []

    for enemy_id in range(1, 9):  # Enemies 1 to 8
        env.update_parameter('enemies', [enemy_id])
        _, player_energy, enemy_energy, _ = env.play(pcont=best_weights)
        player_energy_list.append(player_energy)
        enemy_energy_list.append(enemy_energy)
    
    return player_energy_list, enemy_energy_list

# Calculate energy points for DE Group 1
de_player_energy_group1, de_enemy_energy_group1 = calculate_energy_points(env_group1_de, best_solution_group1_de)

# Calculate energy points for DE Group 2
de_player_energy_group2, de_enemy_energy_group2 = calculate_energy_points(env_group2_de, best_solution_group2_de)

# Store the results in a file or print them for reference
print("DE Group 1 - Player Energy:", de_player_energy_group1)
print("DE Group 1 - Enemy Energy:", de_enemy_energy_group1)
print("DE Group 2 - Player Energy:", de_player_energy_group2)
print("DE Group 2 - Enemy Energy:", de_enemy_energy_group2)

# You can also save the results to a file if needed
np.savetxt('de_player_energy_group1.txt', de_player_energy_group1)
np.savetxt('de_enemy_energy_group1.txt', de_enemy_energy_group1)
np.savetxt('de_player_energy_group2.txt', de_player_energy_group2)
np.savetxt('de_enemy_energy_group2.txt', de_enemy_energy_group2)
