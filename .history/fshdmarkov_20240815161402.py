import numpy as np
from scipy.stats import dirichlet
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import random

# Initial and observed state distributions
initial_state_distribution = {"S": 5488, "E": 0, "I": 0, "R": 0, "D": 0}
observed_state_distribution_3days = {"S": 4956, "E": 14, "I": 13, "R": 150, "D": 355}

# Function to simulate Markov model over time
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

# Function to create transition probability matrix from rates
def create_transition_matrix(Δ, Dr, VD, d0, VT, TD):
    transition_probabilities_hourly = {
        "S": np.array([1 - (VD + Δ), VD, 0, Δ, 0]),
        "E": np.array([d0, 1 - (d0 + VT * TD + Δ), VT * TD + Δ, 0, 0]),
        "I": np.array([0, 0, 1 - (VD + Dr), 0, Dr]),
        "R": np.array([0, 0, VD, 1 - (VD + Dr), Dr]),
        "D": np.array([0, 0, 0, 0, 1.0])
    }
    return transition_probabilities_hourly

# Sum of squared residuals (SSR) function
def SSR_Score(predicted_distribution, observed_distribution):
    score = sum((predicted_distribution[state] - observed_distribution[state]) ** 2 for state in predicted_distribution)
    return score

# Objective function for optimization
def objective_function(params):
    Δ, Dr, VD, d0, VT, TD = params
    transition_probabilities = create_transition_matrix(Δ, Dr, VD, d0, VT, TD)
    simulation_history = simulate_markov_model(transition_probabilities, initial_state_distribution, 72)  # 72 hours = 3 days
    final_distribution = simulation_history[-1]
    score = SSR_Score(final_distribution, observed_state_distribution_3days)
    return score

# Initial guess for the parameters
initial_guess = [0.01, 1/20.1, 0.00211, 0.246, 6.41, 1/13]

# Run optimization
result = minimize(objective_function, initial_guess, method='Nelder-Mead')

# Extract optimized parameters
optimized_params = result.x
optimized_transition_probabilities = create_transition_matrix(*optimized_params)
simulation_history = simulate_markov_model(optimized_transition_probabilities, initial_state_distribution, 72)

print("Optimized Transition Rates:")
print(f"Δ: {optimized_params[0]}")
print(f"Dr: {optimized_params[1]}")
print(f"VD: {optimized_params[2]}")
print(f"d0: {optimized_params[3]}")
print(f"VT: {optimized_params[4]}")
print(f"TD: {optimized_params[5]}")

# Plot the cell states over time
time_points = range(len(simulation_history))
states = list(simulation_history[0].keys())

plt.figure(figsize=(10, 6))

for state in states:
    state_counts = [distribution[state] for distribution in simulation_history]
    plt.plot(time_points, state_counts, label=state)

plt.xlabel('Time (hours)')
plt.ylabel('Number of Cells')
plt.title('Cell States Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Validate the final state distribution with observed data at 72 hours (3 days)
final_distribution = simulation_history[-1]
observed_distribution = observed_state_distribution_3days

print("\nFinal simulated distribution vs. observed distribution (72 hours):")
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
ax.set_title('Final State Distribution at 72 Hours')
ax.set_xticks(x)
ax.set_xticklabels(states)
ax.legend()

plt.show()
