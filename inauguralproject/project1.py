import numpy as np
import matplotlib.pyplot as plt

# Given parameters
alpha = 1/3
beta = 2/3
omega_A = np.array([0.8, 0.3])
omega_B = np.array([1.0, 1.0]) - omega_A  # Total endowment is 1 for each good

# Utility functions for A and B
def utility_A(x1, x2, alpha=alpha):
    return (x1**alpha) * (x2**(1-alpha))

def utility_B(x1, x2, beta=beta):
    return (x1**beta) * (x2**(1-beta))

# Initial utility levels with initial endowments
initial_utility_A = utility_A(omega_A[0], omega_A[1])
initial_utility_B = utility_B(omega_B[0], omega_B[1])

# Define the range for x1_A and x2_A
N = 75
x1_A = np.linspace(0, 1, N)
x2_A = np.linspace(0, 1, N)
x1_A_grid, x2_A_grid = np.meshgrid(x1_A, x2_A)

# Calculate utilities for all possible combinations in the Edgeworth box
utility_A_grid = utility_A(x1_A_grid, x2_A_grid)
utility_B_grid = utility_B(1 - x1_A_grid, 1 - x2_A_grid)

# Finding the Pareto optimal allocations
pareto_optimal_allocations = (utility_A_grid >= initial_utility_A) & (utility_B_grid >= initial_utility_B)

# Plotting the Edgeworth Box
plt.figure(figsize=(8, 8))
plt.plot(omega_A[0], omega_A[1], 'ro', label='Initial Endowment A')
plt.plot(omega_B[0], omega_B[1], 'bo', label='Initial Endowment B')
plt.contourf(x1_A_grid, x2_A_grid, pareto_optimal_allocations, levels=[False, True], colors=['blue'], alpha=0.3)
plt.title('Edgeworth Box for Pareto Optimal Allocations')
plt.xlabel('Good 1 for Consumer A')
plt.ylabel('Good 2 for Consumer A')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.show()
