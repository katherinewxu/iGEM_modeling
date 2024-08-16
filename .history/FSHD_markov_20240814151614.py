import numpy as np
from scipy.stats import dirichlet, chisquare
import matplotlib.pyplot as plt

# Initial and observed state distributions
initial_state_distribution = {"S": 5488, "E": 0, "I": 0, "R": 0, "D": 0}
observed_state_distribution_3days = {"S": 4956, "E": 14, "I": 13, "R": 150, "D": 355}

# Transition rates (preserving zeros)
transition_probabilities = {
    "S": np.array([0.0, 1, 0.0, 0.0, 0.0]),  
    "E": np.array([0.33, 0.0, 0.667, 0.0, 0.0]),  
    "I": np.array([0.0, 0.0, 0.0, 0.041, 0.959]), 
    "R": np.array([0.0, 0.0, 0.832, 0.0, 0.168])
}

# Normalize non-zero probabilities
for state, probabilities in transition_probabilities.items():
    non_zero_indices = probabilities > 0
    sum_non_zero = np.sum(probabilities[non_zero_indices])
    if sum_non_zero > 0:
        transition_probabilities[state][non_zero_indices] /= sum_non_zero

# Function to simulate Markov model
def simulate_markov_model(transition_probabilities, initial_state_distribution, time_steps):
    state_distribution = initial_state_distribution.copy()
    states = list(initial_state_distribution.keys())
    history = [state_distribution.copy()]
    
    for _ in range(time_steps):
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

# Bayesian Optimization
def bayesian_optimization(transition_probabilities, initial_state_distribution, observed_distribution, iterations=10000):
    best_probabilities = None
    best_score = float('inf')

    for _ in range(iterations):
        sampled_probabilities = {}
        for state, probs in transition_probabilities.items():
            if np.any(probs > 0):  # Only sample from Dirichlet if there are non-zero probabilities
                non_zero_indices = probs > 0
                sampled_probs = dirichlet.rvs(probs[non_zero_indices])[0]
                new_probs = np.zeros_like(probs)
                new_probs[non_zero_indices] = sampled_probs
                sampled_probabilities[state] = new_probs
            else:
                sampled_probabilities[state] = probs
        
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

# Optimize the transition probabilities using Bayesian Optimization
optimized_probabilities = bayesian_optimization(transition_probabilities, initial_state_distribution, observed_state_distribution_3days)
simulation_history = simulate_markov_model(optimized_probabilities, initial_state_distribution, 3)

print("Optimized Transition Probabilities:")
print(optimized_probabilities)

# Print the history of state distributions
for t, distribution in enumerate(simulation_history):
    print(f"Day {t} Cell States: {distribution}")

# Validate the final state distribution with observed data at 3 days
final_distribution = simulation_history[-1]
observed_distribution = observed_state_distribution_3days

print("\nFinal simulated distribution vs. observed distribution (3 days):")
for state in final_distribution:
    print(f"{state}: Simulated={final_distribution[state]}, Observed={observed_distribution[state]}")

# Chi-squared test for goodness of fit
observed_counts = np.array(list(observed_distribution.values()))
predicted_counts = np.array(list(final_distribution.values()))

# Avoid division by zero by removing zero elements
non_zero_mask = (predicted_counts != 0) & (observed_counts != 0)
observed_counts = observed_counts[non_zero_mask]
predicted_counts = predicted_counts[non_zero_mask]

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
