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
with open('results/results_LetterEnv_numerical_RML_simple_encoding_up_to_10.pkl', 'rb') as f:
    RML_results = pickle.load(f)

with open('results/results_LetterEnv_numerical_CRA_200_run.pkl', 'rb') as f:
    CQL_results = pickle.load(f)

with open('results/results_LetterEnv_numerical_CRA_no_counterfactual.pkl', 'rb') as f:
    CRA_results = pickle.load(f)

# Print the data to confirm structure (just for inspection)
print(RML_results)
print(CRA_results)
print('CQL - ', CQL_results)

# Extract mean and standard deviation for RML_results (pandas DataFrame)
rml_mean = RML_results.groupby('n value')[['steps']].mean()
rml_std = RML_results.groupby('n value')[['steps']].std()

# Convert RML data to match CRA's format (lists for means and stds)
rml_means = rml_mean['steps'].tolist()
rml_stds = rml_std['steps'].tolist()
rml_n_values = rml_mean.index.tolist()

# Extract mean and standard deviation for CRA_results (dictionary)
cra_n_values = list(CRA_results.keys())
cra_means = [np.mean(CRA_results[n]) for n in cra_n_values]
cra_stds = [np.std(CRA_results[n]) for n in cra_n_values]

cql_n_values = list(CQL_results.keys())
cql_means = [np.mean(CQL_results[n]) for n in cql_n_values]
cql_stds = [np.std(CQL_results[n]) for n in cql_n_values]

# Create the plot
fig, axes = plt.subplots(1, 1, figsize=(10, 6))

# Plot 1: Mean of Steps with Standard Deviation fill
axes.plot(rml_n_values, rml_means, label='RML', color='y', marker='o')
axes.fill_between(rml_n_values, np.array(rml_means) - np.array(rml_stds), np.array(rml_means) + np.array(rml_stds), color='y', alpha=0.2)

axes.plot(cra_n_values, cra_means, label='QL', color='r', marker='x')
axes.fill_between(cra_n_values, np.array(cra_means) - np.array(cra_stds), np.array(cra_means) + np.array(cra_stds), color='r', alpha=0.2)

axes.plot(cql_n_values, cql_means, label='CQL', color='b', marker='x')
axes.fill_between(cql_n_values, np.array(cql_means) - np.array(cql_stds), np.array(cql_means) + np.array(cql_stds), color='b', alpha=0.2)

# Highlight the area for n = 4 and n = 5 where CRA can't handle
axes.axvline(x=3, color='gray', linestyle='--')

# Add titles and labels
#axes.set_title('Mean and Std Deviation of Steps')
axes.set_xlabel('N value')
axes.set_ylabel('Steps')

# Add legends
axes.legend()

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
