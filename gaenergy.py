import os
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller

n_hidden_neurons = 10

# Load best solutions for GA
best_solution_group1 = np.loadtxt('group1_demo_ga/best_solution_group1.txt')
best_solution_group2 = np.loadtxt('group2_demo_ga/best_solution_group2.txt')

# Ensure directories exist for logging
if not os.path.exists('ga_test_group1'):
    os.makedirs('ga_test_group1')

if not os.path.exists('ga_test_group2'):
    os.makedirs('ga_test_group2')

# Create environments
env_group1 = Environment(experiment_name='ga_test_group1', enemies=[1], multiplemode="no", 
                         playermode="ai", player_controller=player_controller(n_hidden_neurons), 
                         enemymode="static", level=2, speed="fastest", visuals=False)

env_group2 = Environment(experiment_name='ga_test_group2', enemies=[1], multiplemode="no", 
                         playermode="ai", player_controller=player_controller(n_hidden_neurons), 
                         enemymode="static", level=2, speed="fastest", visuals=False)

# Function to calculate energy points
def calculate_energy_points(env, best_weights):
    player_energy_list = []
    enemy_energy_list = []

    for enemy_id in range(1, 9):  # Enemies 1 to 8
        env.update_parameter('enemies', [enemy_id])
        _, player_energy, enemy_energy, _ = env.play(pcont=best_weights)
        player_energy_list.append(player_energy)
        enemy_energy_list.append(enemy_energy)
    
    return player_energy_list, enemy_energy_list

# Calculate energy points for Group 1
ga_player_energy_group1, ga_enemy_energy_group1 = calculate_energy_points(env_group1, best_solution_group1)

# Calculate energy points for Group 2
ga_player_energy_group2, ga_enemy_energy_group2 = calculate_energy_points(env_group2, best_solution_group2)

# Store the results in a file or print them for reference
print("GA Group 1 - Player Energy:", ga_player_energy_group1)
print("GA Group 1 - Enemy Energy:", ga_enemy_energy_group1)
print("GA Group 2 - Player Energy:", ga_player_energy_group2)
print("GA Group 2 - Enemy Energy:", ga_enemy_energy_group2)

# You can also save the results to a file if needed
np.savetxt('ga_player_energy_group1.txt', ga_player_energy_group1)
np.savetxt('ga_enemy_energy_group1.txt', ga_enemy_energy_group1)
np.savetxt('ga_player_energy_group2.txt', ga_player_energy_group2)
np.savetxt('ga_enemy_energy_group2.txt', ga_enemy_energy_group2)
