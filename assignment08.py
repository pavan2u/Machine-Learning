import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


# Summation Unit
def summation_unit(inputs, weights=None):
    if weights:
        return sum(i * w for i, w in zip(inputs, weights))
    return sum(inputs)

# Activation Functions
def step_function(x):
    return 1 if x >= 0 else 0

def bipolar_step_function(x):
    return 1 if x >= 0 else -1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Using numpy to avoid overflow

def tanh(x):
    return np.tanh(x)

def relu(x):
    return max(0, x)

def leaky_relu(x, alpha=0.01):
    return x if x >= 0 else alpha * x

# Error Calculation
def mean_squared_error(predicted, actual):
    if len(predicted) != len(actual):
        raise ValueError("Inputs must have the same length.")
    return np.mean((np.array(predicted) - np.array(actual)) ** 2)

# Perceptron Learning Algorithm
def perceptron_learning(X, y, weights, learning_rate, max_epochs=100):
    epochs = 0
    errors = []
    
    while epochs < max_epochs:
        total_error = 0
        for i in range(len(X)):
            inputs = [1] + X[i]  # Adding bias term
            weighted_sum = summation_unit(inputs, weights)
            output = step_function(weighted_sum)
            error = y[i] - output
            total_error += error ** 2  # Sum squared error
            
            for j in range(len(weights)):
                weights[j] += learning_rate * error * inputs[j]
        
        errors.append(total_error)
        epochs += 1
        
        if total_error == 0:
            break
    
    return weights, epochs, errors

# Training with Activation Function
def train_with_activation(X, y, activation_function, learning_rate=0.1, max_epochs=1000):
    weights = np.random.randn(len(X[0]) + 1)
    epochs = 0
    errors = []
    
    while epochs < max_epochs:
        total_error = 0
        for i in range(len(X)):
            inputs = np.insert(X[i], 0, 1)  # Adding bias
            weighted_sum = np.dot(weights, inputs)
            output = activation_function(weighted_sum)
            error = y[i] - output
            total_error += error ** 2
            weights += learning_rate * error * inputs
        
        errors.append(total_error)
        epochs += 1
        
        if total_error == 0:
            break
    
    return weights, epochs, errors

# Testing Summation Unit & Activation Functions
inputs = [1, 2, 3, 4]
print("Summation Unit:", summation_unit(inputs))

x = 0.5
print("Step Function:", step_function(x))
print("Bipolar Step Function:", bipolar_step_function(x))
print("Sigmoid Function:", sigmoid(x))
print("TanH Function:", tanh(x))
print("ReLU Function:", relu(x))
print("Leaky ReLU Function:", leaky_relu(x))

# Mean Squared Error Test
predicted = [3, 5, 2.5, 7]
actual = [3, 5, 2, 8]
print("Mean Squared Error:", mean_squared_error(predicted, actual))

# Training for AND Gate
X_and = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_and = [0, 0, 0, 1]

weights = [10, 0.2, -0.75]
learning_rate = 0.05
final_weights, num_epochs, error_values = perceptron_learning(X_and, y_and, weights, learning_rate, max_epochs=100)

plt.plot(range(1, num_epochs + 1), error_values, marker='o', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Sum of Squared Errors")
plt.title("Error vs. Epochs in Perceptron Learning for AND Gate")
plt.grid()
plt.show()

print("Final Weights:", final_weights)
print("Number of Epochs for Convergence:", num_epochs)

# Training for Customer Transaction Classification
customer_data = np.array([
    [20, 6, 2, 386],
    [16, 3, 6, 289],
    [27, 6, 2, 393],
    [19, 1, 2, 110],
    [24, 4, 2, 280],
    [22, 1, 5, 167],
    [15, 4, 2, 271],
    [18, 4, 2, 274],
    [21, 1, 4, 148],
    [16, 2, 4, 198]
])

y_values = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])  # High Value Tx? (1 = Yes, 0 = No)

final_w, epochs, err_vals = train_with_activation(customer_data, y_values, sigmoid, learning_rate=0.1, max_epochs=1000)

print("Customer Transaction Classification - Final Weights:", final_w)
print("Epochs for Convergence:", epochs)

plt.plot(range(1, epochs + 1), err_vals, marker='o', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Sum of Squared Errors")
plt.title("Error vs. Epochs for Customer Transaction Classification")
plt.grid()
plt.show()

# Comparison with Matrix Pseudo-Inverse
X_bias = np.hstack((np.ones((customer_data.shape[0], 1)), customer_data))  # Add bias column
pseudo_inverse_weights = np.linalg.pinv(X_bias).dot(y_values)

print("Pseudo-Inverse Weights:", pseudo_inverse_weights)

# Neural Network for AND Gate with Backpropagation
class NeuralNetwork:
    def __init__(self, learning_rate=0.05):
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.rand(2, 2)
        self.weights_hidden_output = np.random.rand(2)
        self.bias_hidden = np.random.rand(2)
        self.bias_output = np.random.rand(1)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def train(self, X, y, max_epochs=1000, error_threshold=0.002):
        for epoch in range(max_epochs):
            total_error = 0
            for i in range(len(X)):
                hidden_input = np.dot(X[i], self.weights_input_hidden) + self.bias_hidden
                hidden_output = self.sigmoid(hidden_input)
                final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
                final_output = self.sigmoid(final_input)
                
                error = y[i] - final_output
                total_error += error ** 2
                
                d_output = error * self.sigmoid_derivative(final_output)
                d_hidden = d_output * self.weights_hidden_output * self.sigmoid_derivative(hidden_output)
                
                self.weights_hidden_output += self.learning_rate * d_output * hidden_output
                self.weights_input_hidden += self.learning_rate * np.outer(X[i], d_hidden)
            
            if total_error <= error_threshold:
                break
        
        return epoch

nn = NeuralNetwork()
nn.train(X_and, y_and)
print("Neural Network trained for AND Gate with Backpropagation.")

# Training for XOR Gate
X_xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_xor = [0, 1, 1, 0]

weights = [10, 0.2, -0.75]
learning_rate = 0.05
final_weights, num_epochs, error_values = perceptron_learning(X_xor, y_xor, weights, learning_rate, max_epochs=100)

plt.plot(range(1, num_epochs + 1), error_values, marker='o', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Sum of Squared Errors")
plt.title("Error vs. Epochs in Perceptron Learning for AND Gate")
plt.grid()
plt.show()

print("Final Weights:", final_weights)
print("Number of Epochs for Convergence:", num_epochs)

def summation_unit(inputs, weights):
    return sum(x * w for x, w in zip(inputs + [1], weights))  # Including bias

def step_activation(z):
    return 1 if z >= 0 else 0

def error_comparator(actual, target):
    return actual - target

class Perceptron:
    def __init__(self, input_size, output_size, learning_rate):
        self.weights = [10, 0.2, -0.75] * output_size  # Initial weights (W0, W1, W2) for each output node
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

    def activation_function(self, z):
        return step_activation(z)

    def predict(self, inputs):
        outputs = []
        for i in range(self.output_size):
            weighted_sum = summation_unit(inputs, self.weights[i*self.input_size:(i+1)*self.input_size])
            outputs.append(self.activation_function(weighted_sum))
        return outputs

    def train(self, training_inputs, targets, epochs):
        for epoch in range(epochs):
            for inputs, target in zip(training_inputs, targets):
                predicted_outputs = self.predict(inputs)
                errors = [error_comparator(predicted_outputs[i], target[i]) for i in range(self.output_size)]
                for i in range(self.output_size):
                    for j in range(self.input_size):
                        self.weights[i*self.input_size + j] -= self.learning_rate * errors[i] * inputs[j]

# Training data for AND gate with 2 output nodes
training_inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

targets = [
    [1, 0],  # 0 AND 0 -> [1, 0]
    [0, 1],  # 0 AND 1 -> [0, 1]
    [0, 1],  # 1 AND 0 -> [0, 1]
    [0, 1]   # 1 AND 1 -> [0, 1]
]

perceptron = Perceptron(input_size=2, output_size=2, learning_rate=0.05)
perceptron.train(training_inputs, targets, epochs=1000)

print("Testing AND gate with 2 output nodes:")
for inputs in training_inputs:
    outputs = perceptron.predict(inputs)
    print(f"Input: {inputs} -> Output: {outputs}")

# Implementing MLPClassifier for AND and XOR gates
mlp_and = MLPClassifier(hidden_layer_sizes=(2,), activation='relu', max_iter=1000)
mlp_xor = MLPClassifier(hidden_layer_sizes=(2,), activation='relu', max_iter=1000)

# Training data and expected outputs for AND gate
and_targets = [0, 0, 0, 1]  # Standard AND gate outputs
mlp_and.fit(training_inputs, and_targets)

print("\nMLPClassifier AND gate predictions:")
for inputs in training_inputs:
    print(f"Input: {inputs} -> Output: {mlp_and.predict([inputs])}")

# Training data and expected outputs for XOR gate
xor_targets = [0, 1, 1, 0]  # Standard XOR gate outputs
mlp_xor.fit(training_inputs, xor_targets)

print("\nMLPClassifier XOR gate predictions:")
for inputs in training_inputs:
    print(f"Input: {inputs} -> Output: {mlp_xor.predict([inputs])}")