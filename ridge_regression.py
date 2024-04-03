import numpy as np
from scipy.optimize import minimize

# Generate some sample data
X = np.random.normal(size=(100, 10))
y = np.random.normal(size=(100,))

# Define the objective function
def objective(beta, X, y, lamb):
    return np.sum((y - X.dot(beta)) ** 2) + lamb * np.sum(beta ** 2)

# Define the constraint function
def constraint(beta, X, y, T):
    return T - np.sqrt(np.sum(beta ** 2))

# Set the initial values for the betas and the regularization parameter
beta_init = np.zeros(10)
lamb = 0.1
T = 1.0

# Define the optimization problem
problem = {'fun': objective, 'x0': beta_init, 'args': (X, y, lamb), 'constraints': {'type': 'ineq', 'fun': constraint, 'args': (X, y, T)}}

# Solve the optimization problem
result = minimize(**problem)

# Print the results
print('beta:', result.x)
print('objective:', result.fun)
print('constraint:', T - np.sqrt(np.sum(result.x ** 2)))