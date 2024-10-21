import numpy as np
import matplotlib.pyplot as plt
import os

# Define the directory paths for GA and DE results
ga_directory = 'results_ga'
de_directory = 'results_de'

# Number of generations (this should match the length of the fitness logs)
generations = np.arange(1, 31)  # assuming 100 generations

def load_ga_fitness_data(env_num):
    mean_fitness_list = []
    max_fitness_list = []

    # Loop over the 10 runs for GA
    for run in range(1, 11):  # assuming 10 runs
        mean_file = f'ga_mean_fitness_env{env_num}_run{run}.txt'
        max_file = f'ga_max_fitness_env{env_num}_run{run}.txt'

        mean_fitness_list.append(np.loadtxt(os.path.join(ga_directory, mean_file)))
        max_fitness_list.append(np.loadtxt(os.path.join(ga_directory, max_file)))

    # Convert lists to arrays and compute mean/std across the 10 runs
    mean_fitness_avg = np.mean(mean_fitness_list, axis=0)
    max_fitness_avg = np.mean(max_fitness_list, axis=0)

    mean_fitness_std = np.std(mean_fitness_list, axis=0)
    max_fitness_std = np.std(max_fitness_list, axis=0)

    return mean_fitness_avg, max_fitness_avg, mean_fitness_std, max_fitness_std


def load_de_fitness_data(group_num):
    # DE doesn't have separate files for different runs, instead we load avg and std directly.
    mean_fitness_file = f'g{group_num}mean_fitness_avg.txt'
    max_fitness_file = f'g{group_num}max_fitness_avg.txt'
    mean_fitness_std_file = f'g{group_num}mean_fitness_std.txt'
    max_fitness_std_file = f'g{group_num}max_fitness_std.txt'

    mean_fitness_avg = np.loadtxt(os.path.join(de_directory, mean_fitness_file))
    max_fitness_avg = np.loadtxt(os.path.join(de_directory, max_fitness_file))
    mean_fitness_std = np.loadtxt(os.path.join(de_directory, mean_fitness_std_file))
    max_fitness_std = np.loadtxt(os.path.join(de_directory, max_fitness_std_file))

    return mean_fitness_avg, max_fitness_avg, mean_fitness_std, max_fitness_std


# Load GA data for Environment 1 and 2
ga_mean_fitness_env1, ga_max_fitness_env1, ga_mean_fitness_std_env1, ga_max_fitness_std_env1 = load_ga_fitness_data(1)
ga_mean_fitness_env2, ga_max_fitness_env2, ga_mean_fitness_std_env2, ga_max_fitness_std_env2 = load_ga_fitness_data(2)

# Load DE data for Group 1 and Group 2
de_mean_fitness_group1, de_max_fitness_group1, de_mean_fitness_std_group1, de_max_fitness_std_group1 = load_de_fitness_data(1)
de_mean_fitness_group2, de_max_fitness_group2, de_mean_fitness_std_group2, de_max_fitness_std_group2 = load_de_fitness_data(2)

# Plot for GA and DE comparison in Environment 1 / Group 1
plt.figure(figsize=(10, 6))
plt.plot(generations, ga_mean_fitness_env1, label='GA Mean Fitness (Env 1)', color='blue')
plt.fill_between(generations, ga_mean_fitness_env1 - ga_mean_fitness_std_env1, ga_mean_fitness_env1 + ga_mean_fitness_std_env1, color='blue', alpha=0.2)

plt.plot(generations, ga_max_fitness_env1, label='GA Max Fitness (Env 1)', color='blue', linestyle='--')
plt.fill_between(generations, ga_max_fitness_env1 - ga_max_fitness_std_env1, ga_max_fitness_env1 + ga_max_fitness_std_env1, color='blue', alpha=0.2)

plt.plot(generations, de_mean_fitness_group1, label='DE Mean Fitness (Group 1)', color='red')
plt.fill_between(generations, de_mean_fitness_group1 - de_mean_fitness_std_group1, de_mean_fitness_group1 + de_mean_fitness_std_group1, color='red', alpha=0.2)

plt.plot(generations, de_max_fitness_group1, label='DE Max Fitness (Group 1)', color='red', linestyle='--')
plt.fill_between(generations, de_max_fitness_group1 - de_max_fitness_std_group1, de_max_fitness_group1 + de_max_fitness_std_group1, color='red', alpha=0.2)

plt.title('GA vs DE Fitness Progress - Environment 1 / Group 1')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)
plt.show()

# Plot for GA and DE comparison in Environment 2 / Group 2
plt.figure(figsize=(10, 6))
plt.plot(generations, ga_mean_fitness_env2, label='GA Mean Fitness (Env 2)', color='blue')
plt.fill_between(generations, ga_mean_fitness_env2 - ga_mean_fitness_std_env2, ga_mean_fitness_env2 + ga_mean_fitness_std_env2, color='blue', alpha=0.2)

plt.plot(generations, ga_max_fitness_env2, label='GA Max Fitness (Env 2)', color='blue', linestyle='--')
plt.fill_between(generations, ga_max_fitness_env2 - ga_max_fitness_std_env2, ga_max_fitness_env2 + ga_max_fitness_std_env2, color='blue', alpha=0.2)

plt.plot(generations, de_mean_fitness_group2, label='DE Mean Fitness (Group 2)', color='red')
plt.fill_between(generations, de_mean_fitness_group2 - de_mean_fitness_std_group2, de_mean_fitness_group2 + de_mean_fitness_std_group2, color='red', alpha=0.2)

plt.plot(generations, de_max_fitness_group2, label='DE Max Fitness (Group 2)', color='red', linestyle='--')
plt.fill_between(generations, de_max_fitness_group2 - de_max_fitness_std_group2, de_max_fitness_group2 + de_max_fitness_std_group2, color='red', alpha=0.2)

plt.title('GA vs DE Fitness Progress - Environment 2 / Group 2')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)
plt.show()
