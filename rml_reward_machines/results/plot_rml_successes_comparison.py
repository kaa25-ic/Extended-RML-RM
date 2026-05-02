import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set the Seaborn style
sns.set_theme(style="whitegrid")

# Set font sizes
plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 8

# Load the results from the pickle files
with open('results/results_rml_original_final_result.pkl', 'rb') as f:
    RML_original_results = pickle.load(f)

with open('results/letterenv_numerical_rml_rm_rewards.pkl', 'rb') as f:
    RML_rm_results = pickle.load(f)

with open('results/letterenv_numerical_rml_rm_rewards_no_intermediate.pkl', 'rb') as f:
    RML_rm_ni_results = pickle.load(f)

# ---- Params ----
N = 50
SUCCESS = 100
SUCCESS_RML_RM   = 110

# If your pickles are just lists, this is all you need:
orig_rewards = RML_original_results
rm_rewards   = RML_rm_results
rm_ni_rewards = RML_rm_ni_results

orig_success = (np.asarray(orig_rewards) >= SUCCESS).astype(float)
rm_ni_success = (np.asarray(rm_ni_rewards) >= SUCCESS).astype(float)
rm_success   = (np.asarray(rm_rewards)   >= SUCCESS_RML_RM).astype(float)

orig_rate = pd.Series(orig_success).rolling(window=N, min_periods=1).mean().to_numpy()
rm_rate   = pd.Series(rm_success).rolling(window=N, min_periods=1).mean().to_numpy()
rm_ni_rate   = pd.Series(rm_ni_success).rolling(window=N, min_periods=1).mean().to_numpy()


# Plot
episodes = np.arange(1, len(orig_rate) + 1)

plt.figure(figsize=(12, 6))
plt.plot(episodes, orig_rate, label=f"RMLGym")
plt.plot(episodes, rm_rate,   label=f"RML Reward Machines")
plt.plot(episodes, rm_ni_rate,   label=f"RML Reward Machines Without Intermediate Reward")
# plt.title(f"Success Rate Over Last {N} Episodes")
plt.xlabel("Episode")
plt.ylabel("Success rate")
plt.ylim(0, 1.05)
plt.xlim(1, len(episodes))
plt.legend(loc="center right")
plt.tight_layout()
plt.show()

