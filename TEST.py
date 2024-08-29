import numpy as np

# Define the activation functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Define the network architecture
input_neurons = 5
hidden_neurons = 4
output_neurons = 3

# Initialize weights and biases
np.random.seed(42)  # For reproducibility

# Weights between input layer and hidden layer
weights_input_hidden = np.random.randn(input_neurons, hidden_neurons)
bias_hidden = np.random.randn(hidden_neurons)

# Weights between hidden layer and output layer
weights_hidden_output = np.random.randn(hidden_neurons, output_neurons)
bias_output = np.random.randn(output_neurons)

# Sample input (e.g., a batch of 2 samples)
X = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
              [0.5, 0.4, 0.3, 0.2, 0.1]])

# Forward pass through the network
# Input to Hidden Layer
hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
hidden_output = relu(hidden_input)

# Hidden Layer to Output Layer
output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
output = softmax(output_input)

# Print results
print("Input:")
print(X)
print("\nHidden Layer Output (after ReLU):")
print(hidden_output)
print("\nOutput Layer Output (after Softmax):")
print(output)
