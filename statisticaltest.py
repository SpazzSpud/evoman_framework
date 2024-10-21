import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, shapiro

# Load the gain data
gains_ga_group1 = np.loadtxt('gains_group1_ga.txt')
gains_ga_group2 = np.loadtxt('gains_group2_ga.txt')

gains_de_group1 = np.loadtxt('results_de/gains_group1.txt')
gains_de_group2 = np.loadtxt('results_de/gains_group2.txt')

# Perform Shapiro-Wilk test for normality
print("Shapiro-Wilk Test for Normality:")
print("GA Group 1:", shapiro(gains_ga_group1))
print("DE Group 1:", shapiro(gains_de_group1))
print("GA Group 2:", shapiro(gains_ga_group2))
print("DE Group 2:", shapiro(gains_de_group2))

# Check if data is normally distributed
# If p-value < 0.05, we reject the null hypothesis that the data is normally distributed

def perform_statistical_test(ga_data, de_data, group_name):
    # If both groups are normally distributed, use t-test, otherwise use Mann-Whitney U test
    if shapiro(ga_data).pvalue > 0.05 and shapiro(de_data).pvalue > 0.05:
        # Perform t-test (for normally distributed data)
        t_stat, p_val = ttest_ind(ga_data, de_data)
        print(f"T-test result for {group_name}: t-stat={t_stat}, p-value={p_val}")
    else:
        # Perform Mann-Whitney U test (for non-normal data)
        u_stat, p_val = mannwhitneyu(ga_data, de_data)
        print(f"Mann-Whitney U test result for {group_name}: U-stat={u_stat}, p-value={p_val}")

# Perform statistical test for group 1 and group 2
perform_statistical_test(gains_ga_group1, gains_de_group1, "Group 1 (Env 1)")
perform_statistical_test(gains_ga_group2, gains_de_group2, "Group 2 (Env 2)")
