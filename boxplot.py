import matplotlib.pyplot as plt
import numpy as np

# Load gain data for GA and DE from group 1 and group 2
gains_ga_group1 = np.loadtxt('gains_group1_ga.txt')
gains_ga_group2 = np.loadtxt('gains_group2_ga.txt')

gains_de_group1 = np.loadtxt('results_de/gains_group1.txt')  # Assuming DE gains are saved as you previously mentioned
gains_de_group2 = np.loadtxt('results_de/gains_group2.txt')  # Assuming DE gains are saved as you previously mentioned

# Create a figure for boxplots
plt.figure(figsize=(10, 6))

# Create boxplots
data_to_plot = [gains_ga_group1, gains_de_group1, gains_ga_group2, gains_de_group2]

# Create the boxplot
plt.boxplot(data_to_plot, labels=['GA Group 1', 'DE Group 1', 'GA Group 2', 'DE Group 2'])

# Add titles and labels
plt.title('Gain Comparison (GA vs DE)')
plt.ylabel('Gain')
plt.grid(True)

# Show the plot
plt.show()
