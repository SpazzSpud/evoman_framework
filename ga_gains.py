import os
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller

n_hidden_neurons = 10

# Create environment for testing
def create_env(enemies):
    return Environment(
        enemies=enemies,
        multiplemode="no",
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False
    )

# Load the best solution for GA from previous runs (adjust paths as necessary)
best_weights_group1 = np.loadtxt('group1_demo_ga/best_solution_group1.txt')
best_weights_group2 = np.loadtxt('group2_demo_ga/best_solution_group2.txt')

# Function to test the best solution on each enemy (enemies 1 through 8)
def test_best_solution_on_enemies(best_weights, group_name):
    gains = []
    
    # Loop through each enemy (from 1 to 8)
    for enemy in range(1, 9):
        env = create_env([enemy])
        
        # Run the simulation with the best weights
        fitness, player_life, enemy_life, game_time = env.play(pcont=best_weights)
        
        # Calculate the gain (player's life - enemy's life)
        gain = player_life - enemy_life
        gains.append(gain)
        
        print(f"{group_name} - Enemy {enemy}: Gain = {gain}")

    # Save the gains to a text file
    gains_file = f'gains_{group_name}.txt'
    np.savetxt(gains_file, gains)
    print(f"Gains saved to {gains_file}")

# Run the best solution for GA on each enemy in group 1 and 2
test_best_solution_on_enemies(best_weights_group1, "group1_ga")
test_best_solution_on_enemies(best_weights_group2, "group2_ga")
