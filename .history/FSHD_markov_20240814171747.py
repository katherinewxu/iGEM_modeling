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
import numpy as np
from scipy.stats import dirichlet, chisquare
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import dirichlet
import matplotlib.pyplot as plt

# Initial and observed state distributions
initial_state_distribution = np.array([5488, 0, 0, 0, 0])  # S, E, I, R, D
observed_state_distribution_3days = np.array([4956, 14, 13, 150, 355])  # S, E, I, R, D

# Transition probabilities per day
transition_probabilities_daily = np.array([
    [0.0, 1, 0.0, 0.0, 0.0],  # S
    [0.33, 0.0, 0.667, 0.0, 0.0],  # E
    [0.0, 0.0, 0.0, 0.041, 0.959],  # I
    [0.0, 0.0, 0.832, 0.0, 0.168],  # R
    [0.0, 0.0, 0.0, 0.0, 1.0]  # D
])

# Convert daily transition probabilities to hourly
transition_probabilities_hourly = np.zeros_like(transition_probabilities_daily)
for i, probabilities in enumerate(transition_probabilities_daily):
    non_zero_indices = probabilities > 0
    if np.any(non_zero_indices):  # Only adjust non-terminal states
        transition_probabilities_hourly[i][non_zero_indices] = 1 - np.power(1 - probabilities[non_zero_indices], 1/24)

# Normalize non-zero probabilities using vectorized operations
sums = transition_probabilities_hourly.sum(axis=1, keepdims=True)
transition_probabilities_hourly /= sums  # Broadcasting to normalize

# Function to simulate Markov model over hours
def simulate_markov_model(transition_probabilities, initial_state_distribution, time_steps):
    state_distribution = initial_state_distribution.copy()
    history = [state_distribution.copy()]

    for _ in range(time_steps):
        transitions = np.random.multinomial(state_distribution, transition_probabilities)
        state_distribution = transitions.sum(axis=0)
        history.append(state_distribution.copy())

    return history

# Bayesian Optimization
def bayesian_optimization(transition_probabilities, initial_state_distribution, observed_distribution, iterations=10000):
    best_probabilities = None
    best_score = float('inf')

    for _ in range(iterations):
        sampled_probabilities = {}
        for state, probs in enumerate(transition_probabilities):
            if np.any(probs > 0):  # Only sample from Dirichlet if there are non-zero probabilities
                sampled_probs = dirichlet.rvs(probs[probs > 0])[0]
                new_probs = np.zeros_like(probs)
                new_probs[probs > 0] = sampled_probs
                sampled_probabilities[state] = new_probs
            else:
                sampled_probabilities[state] = probs

        simulation_history = simulate_markov_model(sampled_probabilities, initial_state_distribution, 72)  # Simulate over 72 hours
        final_distribution = simulation_history[-1]

        score = SSR_Score(final_distribution, observed_distribution)
        if score < best_score:
            best_score = score
            best_probabilities = sampled_probabilities

    return best_probabilities

def SSR_Score(predicted_distribution, observed_distribution):
    return np.sum((predicted_distribution - observed_distribution) ** 2)

# Optimize the transition probabilities using Bayesian Optimization
optimized_probabilities = bayesian_optimization(transition_probabilities_hourly, initial_state_distribution, observed_state_distribution_3days)
simulation_history = simulate_markov_model(optimized_probabilities, initial_state_distribution, 72)  # Simulate over 72 hours

print("Optimized Transition Probabilities:")
print(optimized_probabilities)

# Print the history of state distributions at each hour
for t, distribution in enumerate(simulation_history):
    print(f"Hour {t} Cell States: {distribution}")

# Validate the final state distribution with observed data at 72 hours (3 days)
final_distribution = simulation_history[-1]

print("\nFinal simulated distribution vs. observed distribution (72 hours):")
for state, value in enumerate(final_distribution):
    print(f"State {state}: Simulated={value}, Observed={observed_state_distribution_3days[state]}")

# Plot the final distributions for comparison
states = ['S', 'E', 'I', 'R', 'D']
x = np.arange(len(states))
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, final_distribution, width, label='Simulated')
bars2 = ax.bar(x + width/2, observed_state_distribution_3days, width, label='Observed')

ax.set_ylabel('Counts')
ax.set_title('Final State Distribution at 72 Hours')
ax.set_xticks(x)
ax.set_xticklabels(states)
ax.legend()

plt.show()
