import numpy as np
from scipy import stats

#N samples
n=1000
#Significance level
alpha = 0.01


mean_spt = 23.5
std_spt = 8.1

mean_ppo = 24.2
std_ppo = 8.4

# Calculate the pooled standard deviation
pooled_std = np.sqrt(((n - 1) * std_spt**2 + (n - 1) * std_ppo**2) / (n + n - 2))

# Calculate the t-statistic
t_stat = (mean_spt - mean_ppo) / (pooled_std * np.sqrt(1/n + 1/n))

# Calculate the degrees of freedom
df = 2 * n - 2

# Calculate the p-value
p_value = stats.t.sf(np.abs(t_stat), df) * 2  # two-tailed p-value



# Compare p-value with significance level
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the two policies.")
    #print percentage gain
    print(f"Percentage gain: {(mean_spt - mean_ppo) / mean_ppo * 100}%")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the two policies.")
    print(f"Percentage gain: {(mean_spt - mean_ppo) / mean_ppo * 100}%")