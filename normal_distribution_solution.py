#!/usr/bin/env python3
"""
Solution to Exercise 4.11, Question 6 about normal distribution quantiles
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Part (a): Find the value of Φ^(-1)(0.975)
# We know that Φ^(-1)(0.025) = -1.96

# Using symmetry property of the standard normal distribution
inverse_cdf_0975 = -stats.norm.ppf(0.025)
print(f"(a) Φ^(-1)(0.975) = {inverse_cdf_0975}")

# Verify using scipy's built-in function
direct_calculation = stats.norm.ppf(0.975)
print(f"    Direct calculation: {direct_calculation}")

# Explanation of why Φ^(-1)(0.975) = 1.96
print("\nExplanation:")
print("The standard normal distribution is symmetric around 0.")
print("This means that Φ(x) + Φ(-x) = 1 for any x.")
print("Therefore, Φ^(-1)(1-p) = -Φ^(-1)(p) for any probability p.")
print("Since Φ^(-1)(0.025) = -1.96, we have Φ^(-1)(0.975) = -Φ^(-1)(0.025) = -(-1.96) = 1.96")

# Part (b): Find the 0.975-quantile of the normal distribution with mean -1 and variance 2
# For a normal distribution with mean μ and standard deviation σ:
# If Z ~ N(0,1) and X ~ N(μ,σ²), then X = μ + σZ
# So the p-quantile of X is: μ + σ * (p-quantile of Z)

mean = -1
variance = 2
std_dev = np.sqrt(variance)  # Standard deviation = sqrt(variance)

# Calculate the 0.975-quantile
quantile_0975 = mean + std_dev * inverse_cdf_0975
print(f"\n(b) 0.975-quantile of N(-1, 2) = {quantile_0975}")

# Verify using scipy's built-in function
direct_calculation_b = stats.norm.ppf(0.975, loc=mean, scale=std_dev)
print(f"    Direct calculation: {direct_calculation_b}")

# Explanation
print("\nExplanation:")
print("For a normal distribution with mean μ and standard deviation σ:")
print("If Z ~ N(0,1) and X ~ N(μ,σ²), then X = μ + σZ")
print("So the p-quantile of X is: μ + σ * (p-quantile of Z)")
print(f"In this case: -1 + √2 * 1.96 = -1 + {std_dev:.4f} * 1.96 = {quantile_0975:.4f}")

# Visualization to illustrate the solutions
plt.figure(figsize=(12, 6))

# Plot for part (a)
plt.subplot(1, 2, 1)
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x)
plt.plot(x, y, 'b-', linewidth=2)
plt.fill_between(x, 0, y, where=(x <= -1.96), color='red', alpha=0.3)
plt.fill_between(x, 0, y, where=(x >= 1.96), color='green', alpha=0.3)
plt.axvline(x=-1.96, color='r', linestyle='--', label='Φ^(-1)(0.025) = -1.96')
plt.axvline(x=1.96, color='g', linestyle='--', label='Φ^(-1)(0.975) = 1.96')
plt.title('Standard Normal Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

# Plot for part (b)
plt.subplot(1, 2, 2)
x = np.linspace(-6, 4, 1000)
y = stats.norm.pdf(x, loc=mean, scale=std_dev)
plt.plot(x, y, 'b-', linewidth=2)
plt.fill_between(x, 0, y, where=(x <= quantile_0975), color='blue', alpha=0.3)
plt.axvline(x=quantile_0975, color='r', linestyle='--', 
            label=f'0.975-quantile = {quantile_0975:.4f}')
plt.title('Normal Distribution N(-1, 2)')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('normal_quantiles.png')
plt.close()

print("\nVisualization saved as 'normal_quantiles.png'")
