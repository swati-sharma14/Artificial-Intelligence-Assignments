import numpy as np

# Step 1: Compute the gradients with respect to m and b
def gradients(m, b, x, y):
    n = len(x)
    dm = -(2/n) * np.sum(x * (y - (m * x + b)))
    db = -(2/n) * np.sum(y - (m * x + b))
    return dm, db

# Step 2: Update the parameters (m and b) using gradient descent
def gradient_descent(x, y, learning_rate, iterations):
    m, b = 0, 0

    for _ in range(iterations):
        dm, db = gradients(m, b, x, y)
        m -= learning_rate * dm
        b -= learning_rate * db

    return m, b

# Generate synthetic data
np.random.seed(42)
X = np.arange(-20, 20, 0.1)
np.random.shuffle(X)
eps = np.random.rand(400) * 10
y = 23 * X + 43 + eps

# Set learning rate and number of iterations
learning_rate = 0.00725
num_iterations = 100

# Perform gradient descent
final_theta = gradient_descent(X, y, learning_rate, num_iterations)

# Print the final values of weights
print(f"w = {final_theta[0]}, b = {final_theta[1]}")
