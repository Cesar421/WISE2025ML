#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
import sys

np.random.seed(42)

##################################################################################
# Lab Class ML:IV
# Exercise 8: P Classification with Neural Networks
##################################################################################

GROUP = "02"  # TODO: write in your group number


def load_feature_vectors(filename: str) -> np.array:
    """
    Load the feature vectors from the dataset in the given file and return
    them as a numpy array with shape (number-of-examples, number-of-features + 1).
    """
    df = pd.read_csv(filename, sep='\t')
    # Drop id column and convert to numpy
    feature_cols = [col for col in df.columns if col != 'id']
    X = df[feature_cols].astype(float).values
    # Add bias term (column of ones) as first column
    ones = np.ones((X.shape[0], 1))
    X_with_bias = np.hstack([ones, X])
    return X_with_bias


def load_class_values(filename: str) -> np.array:
    """
    Load the class values for is_human (class 0 for False and class 1
    for True) from the dataset in the given file and return
    them as a one-dimensional numpy array.
    """
    df = pd.read_csv(filename, sep='\t')
    # Convert True/False to 1/0
    cs = df['is_human'].map({True: 1, False: 0}).values
    return cs


def misclassification_rate(cs: np.array, ys: np.array) -> float:
    """
    This function takes two vectors with gold and predicted labels and
    returns the percentage of positions where truth and prediction disagree
    """
    if len(cs) == 0:
        return float('nan')
    else:
        misclassified = np.sum(cs != ys)
        return misclassified / len(cs)


def encode_class_values(cs: np.array, k: int = 2) -> np.array:  # change for 3 classes
    """
    (a) Encode class values as vectors c ∈ {0, 1}^k (see slide ML:IV-100).
    
    Arguments:
    - cs: one-dimensional numpy array with class values (0, 1, ..., k-1)
    - k: number of classes
    
    Returns:
    - two-dimensional numpy array with shape (n, k) where n is the number of examples
    """
    n = len(cs)
    encoded = np.zeros((n, k))  # change for 3 classes
    for i in range(n):
        encoded[i, cs[i]] = 1
    return encoded


def sigmoid(z: np.array) -> np.array:
    """
    Apply sigmoid activation function element-wise.
    """
    # Clip to avoid overflow
    z_clipped = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z_clipped))


def softmax(z: np.array) -> np.array:
    """
    Apply softmax activation function.
    For numerical stability, subtract max from z.
    """
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def predict_probabilities(Wh: np.array, Wo: np.array, xs: np.array) -> np.array:
    """
    (b) Predict the class probabilities for each example of a dataset.
    
    Arguments:
    - Wh: weight matrix for hidden layer with shape (l, p+1)
    - Wo: weight matrix for output layer with shape (k, l+1)  # change for 3 classes
    - xs: feature vectors with shape (n, p+1)
    
    Returns:
    - class probabilities with shape (n, k)  # change for 3 classes
    """
    # Forward pass through hidden layer
    zh = np.dot(xs, Wh.T)  # (n, l)
    ah = sigmoid(zh)  # (n, l)
    
    # Add bias term to hidden layer activations
    ah_with_bias = np.hstack([np.ones((ah.shape[0], 1)), ah])  # (n, l+1)
    
    # Forward pass through output layer
    zo = np.dot(ah_with_bias, Wo.T)  # (n, k)  # change for 3 classes
    ao = softmax(zo)  # (n, k)  # change for 3 classes
    
    return ao


def predict(Wh: np.array, Wo: np.array, xs: np.array) -> np.array:
    """
    (c) Predict the class for each example of a dataset (as the one with the highest probability).
    
    Arguments:
    - Wh: weight matrix for hidden layer with shape (l, p+1)
    - Wo: weight matrix for output layer with shape (k, l+1)  # change for 3 classes
    - xs: feature vectors with shape (n, p+1)
    
    Returns:
    - predicted class labels with shape (n,)
    """
    probs = predict_probabilities(Wh, Wo, xs)
    return np.argmax(probs, axis=1)


def initialize_random_weights(rows: int, cols: int) -> np.array:
    """
    Generate a pseudorandom weight matrix of dimension (rows, cols).
    """
    return np.random.randn(rows, cols) * 0.01


def train_multilayer_perceptron(xs: np.array, cs: np.array, l: int = 10, 
                                 eta: float = 0.1, iterations: int = 1000, 
                                 validation_fraction: float = 0.2) -> Tuple[np.array, np.array, List[float], List[float], List[Tuple[np.array, np.array]]]:
    """
    (d) Fit a multilayer perceptron using the IGD Algorithm (slide ML:IV-100).
    
    Arguments:
    - xs: feature vectors in the training dataset with shape (n, p+1)
    - cs: class values for every element in xs (values 0, 1, ..., k-1)
    - l: number of hidden units
    - eta: learning rate
    - iterations: number of iterations to run the algorithm
    - validation_fraction: fraction of xs and cs used for validation
    
    Returns:
    - Wh: learned weights for hidden layer with shape (l, p+1)
    - Wo: learned weights for output layer with shape (k, l+1)  # change for 3 classes
    - train_misclass_history: misclassification rates on training set
    - val_misclass_history: misclassification rates on validation set
    - weights_history: list of (Wh, Wo) tuples after each iteration
    """
    n = len(xs)
    p = xs.shape[1]
    k = 2  # number of classes  # change for 3 classes
    
    # Split into training and validation sets
    if validation_fraction > 0:
        n_val = int(n * validation_fraction)
        n_train = n - n_val
        
        # Shuffle indices
        indices = np.random.permutation(n)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        xs_train = xs[train_indices]
        cs_train = cs[train_indices]
        xs_val = xs[val_indices]
        cs_val = cs[val_indices]
    else:
        xs_train = xs
        cs_train = cs
        xs_val = np.array([])
        cs_val = np.array([])
    
    # Encode class values
    cs_train_encoded = encode_class_values(cs_train, k)  # change for 3 classes
    
    # Initialize weights
    Wh = initialize_random_weights(l, p)
    Wo = initialize_random_weights(k, l + 1)  # change for 3 classes
    
    train_misclass_history = []
    val_misclass_history = []
    weights_history = []
    
    for iteration in range(iterations):
        # Forward pass for all training examples
        # Hidden layer
        zh = np.dot(xs_train, Wh.T)  # (n_train, l)
        ah = sigmoid(zh)  # (n_train, l)
        ah_with_bias = np.hstack([np.ones((ah.shape[0], 1)), ah])  # (n_train, l+1)
        
        # Output layer
        zo = np.dot(ah_with_bias, Wo.T)  # (n_train, k)  # change for 3 classes
        ao = softmax(zo)  # (n_train, k)  # change for 3 classes
        
        # Backward pass (compute gradients)
        # Output layer error
        delta_o = ao - cs_train_encoded  # (n_train, k)  # change for 3 classes
        
        # Hidden layer error
        # Remove bias term from Wo for backpropagation
        Wo_no_bias = Wo[:, 1:]  # (k, l)  # change for 3 classes
        delta_h = np.dot(delta_o, Wo_no_bias) * ah * (1 - ah)  # (n_train, l)
        
        # Compute gradients
        grad_Wo = np.dot(delta_o.T, ah_with_bias) / n_train  # (k, l+1)  # change for 3 classes
        grad_Wh = np.dot(delta_h.T, xs_train) / n_train  # (l, p+1)
        
        # Update weights
        Wo = Wo - eta * grad_Wo
        Wh = Wh - eta * grad_Wh
        
        # Store weights
        weights_history.append((Wh.copy(), Wo.copy()))
        
        # Compute misclassification rates
        train_preds = predict(Wh, Wo, xs_train)
        train_misclass = misclassification_rate(cs_train, train_preds)
        train_misclass_history.append(train_misclass)
        
        if len(xs_val) > 0:
            val_preds = predict(Wh, Wo, xs_val)
            val_misclass = misclassification_rate(cs_val, val_preds)
            val_misclass_history.append(val_misclass)
        else:
            val_misclass_history.append(float('nan'))
    
    return Wh, Wo, train_misclass_history, val_misclass_history, weights_history


def plot_learning_curves(train_misclass_history: List[float], 
                         val_misclass_history: List[float],
                         output_file: str = 'learning_curves.png'):
    """
    Plot the learning curves (misclassification rates over iterations).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_misclass_history, label='Training', linewidth=2)
    if not np.isnan(val_misclass_history[0]):
        plt.plot(val_misclass_history, label='Validation', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Misclassification Rate', fontsize=12)
    plt.title('Learning Curves', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Learning curves saved to {output_file}")
    plt.close()


def main():
    if len(sys.argv) != 5:
        print("Usage: python3 programming_exercise_neural_networks.py features-train.tsv labels-train.tsv features-test.tsv predictions-test.tsv")
        sys.exit(1)
    
    features_train_file = sys.argv[1]
    labels_train_file = sys.argv[2]
    features_test_file = sys.argv[3]
    predictions_test_file = sys.argv[4]
    
    # Load training data
    print("Loading training data...")
    xs_train = load_feature_vectors(features_train_file)
    cs_train = load_class_values(labels_train_file)
    
    # Load test data
    print("Loading test data...")
    xs_test = load_feature_vectors(features_test_file)
    
    print(f"Training examples: {len(xs_train)}")
    print(f"Features (including bias): {xs_train.shape[1]}")
    print(f"Test examples: {len(xs_test)}")
    
    # Train multilayer perceptron
    print("\nTraining multilayer perceptron...")
    Wh, Wo, train_misclass_history, val_misclass_history, weights_history = train_multilayer_perceptron(
        xs_train, cs_train,
        l=50,  # number of hidden units
        eta=0.5,  # learning rate
        iterations=1000,
        validation_fraction=0.2
    )
    
    print(f"\nFinal training misclassification rate: {train_misclass_history[-1]:.4f}")
    print(f"Final validation misclassification rate: {val_misclass_history[-1]:.4f}")
    
    # (e) Select the model (iteration) that achieved the best misclassification rate on the validation set
    best_iteration = np.argmin(val_misclass_history)
    best_val_misclass = val_misclass_history[best_iteration]
    print(f"\nBest validation misclassification rate: {best_val_misclass:.4f} at iteration {best_iteration + 1}")
    
    # Use the best model to predict on test set
    Wh_best, Wo_best = weights_history[best_iteration]
    test_predictions = predict(Wh_best, Wo_best, xs_test)
    
    # Convert predictions back to True/False
    test_predictions_bool = [True if p == 1 else False for p in test_predictions]
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'is_human': test_predictions_bool
    })
    predictions_df.to_csv(predictions_test_file, sep='\t', index=False)
    print(f"\nPredictions saved to {predictions_test_file}")
    
    # Plot learning curves
    plot_learning_curves(train_misclass_history, val_misclass_history)


if __name__ == "__main__":
    main()
