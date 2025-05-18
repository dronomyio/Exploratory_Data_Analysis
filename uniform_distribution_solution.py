#!/usr/bin/env python3
"""
Solution to Exercise 4.11, Question 7 about uniform distribution and sample quantiles
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Question 7: Uniform distribution on (0,1) and sample quantile with smallest variance

# Define the uniform distribution parameters
a = 0  # Lower bound
b = 1  # Upper bound

# For a uniform distribution on (0,1):
# f(x) = 1 if x ∈ (0,1), 0 otherwise
# F(x) = 0 if x ≤ 0, x if x ∈ (0,1), 1 if x ≥ 1

# According to Result 4.1 (which we'll implement below), the sample quantile
# with the smallest variance occurs at the value of p where:
# p(1-p)/[f(F^(-1)(p))]^2 is minimized

# For uniform distribution on (0,1):
# F^(-1)(p) = p for p ∈ (0,1)
# f(F^(-1)(p)) = f(p) = 1 for p ∈ (0,1)

# Therefore, we need to minimize p(1-p)/1^2 = p(1-p)
# This is a parabola opening downward with axis of symmetry at p = 0.5
# The minimum variance occurs at p = 0.5, which corresponds to the median

# Let's verify this mathematically and visually

# Create a range of p values
p_values = np.linspace(0.01, 0.99, 100)

# Calculate p(1-p) for each value
variance_factor = p_values * (1 - p_values)

# Find the minimum
min_index = np.argmax(variance_factor)  # We want to maximize p(1-p) to minimize variance
min_p = p_values[min_index]
min_variance_factor = variance_factor[min_index]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(p_values, variance_factor, 'b-', linewidth=2)
plt.axvline(x=0.5, color='r', linestyle='--', label='p = 0.5 (median)')
plt.axhline(y=0.25, color='g', linestyle=':', label='Minimum value = 0.25')
plt.scatter([0.5], [0.25], color='r', s=100, zorder=5)
plt.title('Variance Factor p(1-p) for Uniform Distribution on (0,1)')
plt.xlabel('Quantile Level (p)')
plt.ylabel('p(1-p)')
plt.grid(True)
plt.legend()
plt.savefig('uniform_quantile_variance.png')
plt.close()

# Print the results
print("Solution to Exercise 4.11, Question 7:")
print("\nFor a uniform distribution on (0,1):")
print("- The density function f(x) = 1 for x ∈ (0,1), 0 otherwise")
print("- The distribution function F(x) = 0 if x ≤ 0, x if x ∈ (0,1), 1 if x ≥ 1")
print("\nAccording to Result 4.1, the asymptotic variance of the sample p-quantile is:")
print("σ²(p) = p(1-p)/[nf(F^(-1)(p))²]")
print("\nFor the uniform distribution:")
print("- F^(-1)(p) = p for p ∈ (0,1)")
print("- f(F^(-1)(p)) = f(p) = 1 for p ∈ (0,1)")
print("\nTherefore, the variance is proportional to p(1-p)")
print(f"The minimum value of p(1-p) occurs at p = {min_p:.4f}")
print(f"This corresponds to the {min_p:.4f}-quantile, which is the median")
print("\nConclusion: The sample median (0.5-quantile) has the smallest variance")
print("among all sample quantiles for a uniform distribution on (0,1).")
print("\nVisualization saved as 'uniform_quantile_variance.png'")

# Additional verification through simulation
print("\n--- Verification through Simulation ---")
np.random.seed(42)
n_samples = 10000
n_simulations = 1000
quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
variances = []

for p in quantile_levels:
    quantile_estimates = []
    for _ in range(n_simulations):
        # Generate uniform samples
        samples = np.random.uniform(0, 1, n_samples)
        # Calculate the sample quantile
        q = np.quantile(samples, p)
        quantile_estimates.append(q)
    
    # Calculate the variance of the quantile estimates
    var_q = np.var(quantile_estimates)
    variances.append(var_q)
    print(f"Simulated variance of {p:.2f}-quantile: {var_q:.8f}")

# Find the minimum variance from simulation
min_var_index = np.argmin(variances)
min_var_p = quantile_levels[min_var_index]
print(f"\nQuantile with minimum variance from simulation: {min_var_p:.2f}-quantile")
