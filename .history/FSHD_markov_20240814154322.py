import numpy as np
from scipy.stats import dirichlet, chisquare
import matplotlib.pyplot as plt
import random
# Data retrieval
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix

# # Define the sample file mappings (local paths)
# samples = {
#     "FSHD1.1": "/Users/chris/iGEM_modeling/scRNAseqData/GSM3487556_FSHD1.1.txt",
#     "FSHD1.2": "/Users/chris/iGEM_modeling/scRNAseqData/GSM3487557_FSHD1.2.txt"
#     }

# # Initialize an empty dictionary to store the AnnData objects
# adatas = {}

# # Read and process data for each sample
# for sample_id, filepath in samples.items():
#     try:
#         # Read the text file into a DataFrame 
#         sample_data = pd.read_csv(filepath, sep="\t", index_col=0)
        
#         # Convert the DataFrame to a sparse matrix
#         sample_matrix = csr_matrix(sample_data.values)
#         # Create an AnnData object
#         sample_adata = ad.AnnData(sample_matrix)

#         # Set the observation (cell) names and variable (gene) names
#         sample_adata.var_names_make_unique()  # Ensure gene names are unique
#         sample_adata.obs_names = sample_data.index.tolist()
#         sample_adata.var_names = sample_data.columns.tolist()
        
#         # Store the AnnData object in the dictionary
#         adatas[sample_id] = sample_adata
        
#         print(f"Successfully read data for {sample_id}")
#     except Exception as e:
#         print(f"Failed to read data for {sample_id}: {e}")

# # The adatas dictionary now contains AnnData objects for each sample
# for sample_id, adata in adatas.items():
#     print(f"Data for {sample_id}:")
#     print(adata)  # Print a summary of each AnnData object
#     print(adata.X)  # Print the data matrix

# # Optional: Print the observation and variable names for the first sample
# if adatas:
#     sample_id = list(adatas.keys())[0]
#     adata = adatas[sample_id]
#     print("Observation (cell) names:", adata.obs_names[:10])
#     print("Variable (gene) names:", adata.var_names[:10])

# adata.layers["log_transformed"] = np.log1p(adata.X)
# adata

# adata.to_df(layer="log_transformed")
iimport numpy as np
from scipy.stats import dirichlet, chisquare
import matplotlib.pyplot as plt

# Initial and observed state distributions
initial_state_distribution = {"S": 5488, "E": 0, "I": 0, "R": 0, "D": 0}
observed_state_distribution_3days = {"S": 4956, "E": 14, "I": 13, "R": 150, "D": 355}

# Transition rates (before normalization, preserving zeros)
transition_probabilities = {
    "S": np.array([0.0, 1, 0.0, 0.0, 0.0]),  
    "E": np.array([0.33, 0.0, 0.667, 0.0, 0.0]),  
    "I": np.array([0.0, 0.0, 0.0, 0.041, 0.959]), 
    "R": np.array([0.0, 0.0, 0.832, 0.0, 0.168])
}

# Number of smaller time steps (e.g., if you want to divide into 10 steps)
num_steps = 10

# Adjust transition rates for smaller time steps
adjusted_transition_probabilities = {}
for state, probs in transition_probabilities.items():
    non_zero_indices = probs > 0
    sum_non_zero = np.sum(probs[non_zero_indices])
    
    if sum_non_zero > 0:
        # Scale down for smaller time steps
        scaled_probs = probs / num_steps
        
        # Calculate stay probability
        stay_prob = 1 - np.sum(scaled_probs)
        new_probs = np.insert(scaled_probs, 0, stay_prob)
        
        # Normalize the probabilities again
        adjusted_transition_probabilities[state] = new_probs / np.sum(new_probs)
    else:
        adjusted_transition_probabilities[state] = np.array([1.0])

# Function to simulate Markov model with smaller time steps
def simulate_markov_model(transition_probabilities, initial_state_distribution, time_steps, num_steps):
    state_distribution = initial_state_distribution.copy()
    states = list(initial_state_distribution.keys())
    history = [state_distribution.copy()]
    
    for _ in range(time_steps * num_steps):
        new_distribution = {state: 0 for state in states}
        for state, count in state_distribution.items():
            if count > 0:
                probs = transition_probabilities[state]
                transitions = np.random.multinomial(count, probs)
                for i, next_state in enumerate(states):
                    new_distribution[next_state] += transitions[i]
        state_distribution = new_distribution.copy()
        history.append(state_distribution.copy())
    
    return history

# Simulate with adjusted transition probabilities
simulation_history = simulate_markov_model(adjusted_transition_probabilities, initial_state_distribution, 3, num_steps)

# Print the history of state distributions
for t, distribution in enumerate(simulation_history):
    if t % num_steps == 0:  # Only print after each complete day
        print(f"Day {t // num_steps} Cell States: {distribution}")

# Compare final simulated distribution with observed data at 3 days
final_distribution = simulation_history[-1]
observed_distribution = observed_state_distribution_3days

print("\nFinal simulated distribution vs. observed distribution (3 days):")
for state in final_distribution:
    print(f"{state}: Simulated={final_distribution[state]}, Observed={observed_distribution[state]}")

# Chi-squared test for goodness of fit
observed_counts = np.array(list(observed_distribution.values()))
predicted_counts = np.array(list(final_distribution.values()))

chi2, p_value = chisquare(f_obs=observed_counts, f_exp=predicted_counts)

print(f"\nChi-squared value: {chi2}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("The observed and predicted distributions are significantly different (reject H0).")
else:
    print("The observed and predicted distributions are not significantly different (fail to reject H0).")

# Plot the final distributions for comparison
states = list(final_distribution.keys())
simulated_counts = [final_distribution[state] for state in states]
observed_counts = [observed_distribution[state] for state in states]

x = np.arange(len(states))
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, simulated_counts, width, label='Simulated')
bars2 = ax.bar(x + width/2, observed_counts, width, label='Observed')

ax.set_ylabel('Counts')
ax.set_title('Final State Distribution at 3 Days')
ax.set_xticks(x)
ax.set_xticklabels(states)
ax.legend()

plt.show()
