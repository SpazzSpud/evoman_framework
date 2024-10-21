import numpy as np
import matplotlib.pyplot as plt

# Load the GA mean and max fitness data for Env 1 and Env 2
# Assuming we have 10 runs

mean_fitness_env1_runs = []
max_fitness_env1_runs = []
mean_fitness_env2_runs = []
max_fitness_env2_runs = []

for run in range(1, 11):
    mean_fitness_env1 = np.loadtxt(f'results_ga/ga_mean_fitness_env1_run{run}.txt')
    max_fitness_env1 = np.loadtxt(f'results_ga/ga_max_fitness_env1_run{run}.txt')
    
    mean_fitness_env2 = np.loadtxt(f'results_ga/ga_mean_fitness_env2_run{run}.txt')
    max_fitness_env2 = np.loadtxt(f'results_ga/ga_max_fitness_env2_run{run}.txt')

    mean_fitness_env1_runs.append(mean_fitness_env1)
    max_fitness_env1_runs.append(max_fitness_env1)
    mean_fitness_env2_runs.append(mean_fitness_env2)
    max_fitness_env2_runs.append(max_fitness_env2)

# Convert lists to numpy arrays for easy averaging
mean_fitness_env1_runs = np.array(mean_fitness_env1_runs)
max_fitness_env1_runs = np.array(max_fitness_env1_runs)
mean_fitness_env2_runs = np.array(mean_fitness_env2_runs)
max_fitness_env2_runs = np.array(max_fitness_env2_runs)

# Calculate the average and standard deviation for each generation
mean_fitness_avg_env1 = np.mean(mean_fitness_env1_runs, axis=0)
mean_fitness_std_env1 = np.std(mean_fitness_env1_runs, axis=0)

max_fitness_avg_env1 = np.mean(max_fitness_env1_runs, axis=0)
max_fitness_std_env1 = np.std(max_fitness_env1_runs, axis=0)

mean_fitness_avg_env2 = np.mean(mean_fitness_env2_runs, axis=0)
mean_fitness_std_env2 = np.std(mean_fitness_env2_runs, axis=0)

max_fitness_avg_env2 = np.mean(max_fitness_env2_runs, axis=0)
max_fitness_std_env2 = np.std(max_fitness_env2_runs, axis=0)

# Plot the mean and max fitness with standard deviation for Env 1
generations = np.arange(1, len(mean_fitness_avg_env1) + 1)

plt.figure(figsize=(12, 6))

# Plot for Env 1
plt.subplot(1, 2, 1)
plt.plot(generations, mean_fitness_avg_env1, label='Mean Fitness (Env 1)', color='blue')
plt.fill_between(generations, mean_fitness_avg_env1 - mean_fitness_std_env1, mean_fitness_avg_env1 + mean_fitness_std_env1, color='blue', alpha=0.3)
plt.plot(generations, max_fitness_avg_env1, label='Max Fitness (Env 1)', color='red')
plt.fill_between(generations, max_fitness_avg_env1 - max_fitness_std_env1, max_fitness_avg_env1 + max_fitness_std_env1, color='red', alpha=0.3)
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.title('Mean and Max Fitness with σ for GA (Env 1)')
plt.legend()

# Plot for Env 2
plt.subplot(1, 2, 2)
plt.plot(generations, mean_fitness_avg_env2, label='Mean Fitness (Env 2)', color='blue')
plt.fill_between(generations, mean_fitness_avg_env2 - mean_fitness_std_env2, mean_fitness_avg_env2 + mean_fitness_std_env2, color='blue', alpha=0.3)
plt.plot(generations, max_fitness_avg_env2, label='Max Fitness (Env 2)', color='red')
plt.fill_between(generations, max_fitness_avg_env2 - max_fitness_std_env2, max_fitness_avg_env2 + max_fitness_std_env2, color='red', alpha=0.3)
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.title('Mean and Max Fitness with σ for GA (Env 2)')
plt.legend()

plt.tight_layout()
plt.show()
