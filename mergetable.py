import pandas as pd
import numpy as np

# Load GA and DE gains for Group 1
ga_gains_group1 = np.loadtxt('gains_group1_ga.txt')  # Assuming this file contains the best gains for GA Group 1
de_gains_group1 = np.loadtxt('gains_group1_de.txt')  # Assuming this file contains the best gains for DE Group 1

# Load GA and DE gains for Group 2
ga_gains_group2 = np.loadtxt('gains_group2_ga.txt')  # Assuming this file contains the best gains for GA Group 2
de_gains_group2 = np.loadtxt('gains_group2_de.txt')  # Assuming this file contains the best gains for DE Group 2

# Create a dictionary for combined data
data = {
    'Enemy ID': list(range(1, 9)),  # Enemies from 1 to 8
    'GA Gains (Group 1)': ga_gains_group1,
    'DE Gains (Group 1)': de_gains_group1,
    'GA Gains (Group 2)': ga_gains_group2,
    'DE Gains (Group 2)': de_gains_group2
}

# Convert the dictionary to a pandas DataFrame
gains_df = pd.DataFrame(data)

# Save the merged table to a CSV file for reference
gains_df.to_csv('combined_gains_table.csv', index=False)

# Print the merged table for reference
print(gains_df)
