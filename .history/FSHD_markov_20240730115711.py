import numpy as np
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
import random
# Data retrieval
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix

# Define the sample file mappings (local paths)
samples = {
    "FSHD1.1": "/Users/chris/iGEM_modeling/scRNAseqData/GSM3487556_FSHD1.1.txt",
    "FSHD1.2": "/Users/chris/iGEM_modeling/scRNAseqData/GSM3487557_FSHD1.2.txt"
    }

# Initialize an empty dictionary to store the AnnData objects
adatas = {}

# Read and process data for each sample
for sample_id, filepath in samples.items():
    try:
        # Read the text file into a DataFrame 
        sample_data = pd.read_csv(filepath, sep="\t", index_col=0)
        
        # Convert the DataFrame to a sparse matrix
        sample_matrix = csr_matrix(sample_data.values)
        # Create an AnnData object
        sample_adata = ad.AnnData(sample_matrix)
        # Set the observation (cell) names and variable (gene) names
        sample_adata.obs_names = sample_data.index.tolist()
        sample_adata.var_names = sample_data.columns.tolist()
        
        # Store the AnnData object in the dictionary
        sample_adata.var_names_make_unique()  # Ensure gene names are unique
        adatas[sample_id] = sample_adata
        
        print(f"Successfully read data for {sample_id}")
    except Exception as e:
        print(f"Failed to read data for {sample_id}: {e}")
adata = ad.concat(adatas, label="sample")
adata.obs_names_make_unique()
print(adata.obs["sample"].value_counts())
adata


# Given data for initial and 3-day states
initial_state_distribution = {"S": 5488, "E": 0, "I": 0, "R": 0, "D": 0}
observed_state_distribution_3days = {"S": 4956, "E": 14, "I": 13, "R": 150, "D": 0}

# Define alpha_posterior with small positive values for non-allowed transitions
alpha_posterior = {
    "S": np.array([1, 0.0021, 1e-10, 1e-10, 1e-10]),
    "E": np.array([0.246, 1e-10, 6.41/13, 1e-10, 1e-10]),
    "I": np.array([0.002, 0.002, 1e-10, 0.00211, 1/20.1]),
    "R": np.array([0.002, 0.002, 0.246, 1e-10, 1/20.1])
}

# Function to simulate Markov model
def simulate_markov_model(transition_probabilities, initial_state_distribution, time_steps):
    state_distribution = initial_state_distribution.copy()
    states = list(initial_state_distribution.keys())
    history = [state_distribution.copy()]
    
    for _ in range(time_steps):
        new_distribution = {state: 0 for state in states}
        for state, count in state_distribution.items():
            if count > 0 and state in transition_probabilities:
                probs = transition_probabilities[state]
                transitions = np.random.multinomial(count, probs)
                for i, next_state in enumerate(states):
                    new_distribution[next_state] += transitions[i]
            else:
                new_distribution[state] += count
        state_distribution = new_distribution.copy()
        history.append(state_distribution.copy())
    
    return history

# Bayesian Inference and Optimization
def bayesian_optimization(alpha_posterior, initial_state_distribution, observed_distribution, iterations=100000):
    best_probabilities = None
    best_score = float('inf')

    for _ in range(iterations):
        sampled_probabilities = {}
        for state in alpha_posterior:
            sampled_prob = dirichlet.rvs(alpha_posterior[state])[0]
            sampled_probabilities[state] = sampled_prob
        
        simulation_history = simulate_markov_model(sampled_probabilities, initial_state_distribution, 3)
        final_distribution = simulation_history[-1]
    
        score = SSR_Score(final_distribution, observed_distribution)
        if score < best_score:
            best_score = score
            best_probabilities = sampled_probabilities

    return best_probabilities

def SSR_Score(predicted_distribution, observed_distribution):
    score = sum((predicted_distribution[state] - observed_distribution[state]) ** 2 for state in predicted_distribution)
    return score

optimized_probabilities = bayesian_optimization(alpha_posterior, initial_state_distribution, observed_state_distribution_3days)
simulation_history = simulate_markov_model(optimized_probabilities, initial_state_distribution, 3)

# Print the history of state distributions
for t, distribution in enumerate(simulation_history):
    print(f"Day {t} Cell States: {distribution}")

# Validate the final state distribution with observed data at 3 days
final_distribution = simulation_history[-1]
observed_distribution = observed_state_distribution_3days

print("\nFinal simulated distribution vs. observed distribution (3 days):")
for state in final_distribution:
    print(f"{state}: Simulated={final_distribution[state]}, Observed={observed_distribution[state]}")

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
