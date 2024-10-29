import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
print("Loading data...")
data = pd.read_csv('C:\\Users\\PC\\Desktop\\CI_assignment3\\wdbc.data', header=None)
X = data.iloc[:, 2:].values  # Features
y = np.where(data.iloc[:, 1] == 'M', 1, 0)  # Labels

# Standardization
print("Standardizing data...")
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Cross-validation function
def cross_validation(X, y, hidden_layers, nodes, k=10):
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    splits = np.array_split(indices, k)
    accuracies = []
    
    for i in range(k):
        test_idx = splits[i]
        train_idx = np.concatenate([splits[j] for j in range(k) if j != i])
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Initialize and train MLP model
        model = SimpleMLP(hidden_layers, nodes)
        model.train(X_train, y_train, lr=0.001, epochs=500)
        
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)

    return np.mean(accuracies), np.std(accuracies)

# SimpleMLP class for a basic neural network
class SimpleMLP:
    def __init__(self, hidden_layers, nodes):
        self.hidden_layers = hidden_layers
        self.nodes = nodes
        self.weights = []
        self.biases = []
        
        layer_sizes = [X.shape[1]] + [nodes] * hidden_layers + [1]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, lr=0.001, epochs=500):
        for epoch in range(epochs):
            # Forward pass
            activations = [X]
            for i in range(len(self.weights)):
                net = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                activations.append(self.sigmoid(net))
            
            # Backpropagation
            error = y.reshape(-1, 1) - activations[-1]
            for i in reversed(range(len(self.weights))):
                delta = error * self.sigmoid_derivative(activations[i+1])
                self.weights[i] += lr * np.dot(activations[i].T, delta)
                self.biases[i] += lr * np.sum(delta, axis=0, keepdims=True)
                error = np.dot(delta, self.weights[i].T)

    def predict(self, X):
        for i in range(len(self.weights)):
            X = self.sigmoid(np.dot(X, self.weights[i]) + self.biases[i])
        return (X > 0.5).astype(int).flatten()

# Genetic Algorithm
def genetic_algorithm(population_size, generations):
    print("Starting Genetic Algorithm...")
    population = []
    fitness_history = []
    average_fitness_history = []
    min_fitness_history = []

    for _ in range(population_size):
        hidden_layers = np.random.randint(1, 7)  # 1 to 6 hidden layers
        nodes = np.random.randint(5, 101)  # 5 to 100 nodes
        population.append((hidden_layers, nodes))
    
    for gen in range(generations):
        print(f"Generation {gen + 1}/{generations}...")
        fitness_scores = [cross_validation(X, y, hl, nd)[0] for hl, nd in population]

        fitness_history.append(max(fitness_scores))
        average_fitness_history.append(np.mean(fitness_scores))
        min_fitness_history.append(np.min(fitness_scores))
        
        sorted_population = sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)
        top_half = [x[1] for x in sorted_population[:population_size // 2]]

        new_population = []
        for _ in range(population_size // 2):
            parent_indices = np.random.choice(len(top_half), 2, replace=False)
            parent1_params = top_half[parent_indices[0]]
            parent2_params = top_half[parent_indices[1]]

            child = (parent1_params[0], parent2_params[1])  # Crossover
            if np.random.rand() < 0.1:  # Mutation
                child = (np.random.randint(1, 7), np.random.randint(5, 101))
            new_population.append(child)
        
        population = top_half + new_population
    
    # Plot fitness scores over generations
    plt.plot(fitness_history, label='Best Fitness Score (Accuracy)')
    plt.plot(average_fitness_history, label='Average Fitness Score (Accuracy)')
    plt.plot(min_fitness_history, label='Minimum Fitness Score (Accuracy)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score (Accuracy)')
    plt.title('Fitness Progression Over Generations')
    plt.legend()
    plt.grid()
    plt.show()

    return sorted_population[0][1]  # Return the best parameters

# Run Genetic Algorithm
print("Running Genetic Algorithm...")
best_parameters = genetic_algorithm(population_size=50, generations=50)
print(f"Best parameters found: Hidden Layers: {best_parameters[0]}, Nodes: {best_parameters[1]}")

# Analyze Results
print("Analyzing results...")
final_accuracy, std_dev = cross_validation(X, y, best_parameters[0], best_parameters[1])
print(f"Final Model Accuracy with Best Parameters: {final_accuracy:.2f} ± {std_dev:.2f}")

# Confusion Matrix and Metrics Calculation
print("Calculating confusion matrix and metrics...")
final_model = SimpleMLP(best_parameters[0], best_parameters[1])
final_model.train(X, y)
y_pred = final_model.predict(X)
conf_matrix = np.array([[np.sum((y == 0) & (y_pred == 0)), np.sum((y == 0) & (y_pred == 1))],
                        [np.sum((y == 1) & (y_pred == 0)), np.sum((y == 1) & (y_pred == 1))]])

print("Confusion Matrix:")
print(conf_matrix)

# Additional Metrics: Precision, Recall, F1 Score
precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
f1_score = 2 * (precision * recall) / (precision + recall)

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
