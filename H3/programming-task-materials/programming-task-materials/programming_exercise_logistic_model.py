#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List

np.random.seed(42)

##################################################################################
# Lab Class ML:III
# Starter code for exercise 6: Logistic Model for Generative Authorship Detection
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


def logistic_function(w: np.array, x: np.array) -> float:
    """
    Return the output of a logistic function with parameter vector `w` on
    example `x`.
    Hint: use np.exp(np.clip(..., -30, 30)) instead of np.exp(...) to avoid
    divisions by zero
    """
    z = np.dot(w, x)
    # Clip to avoid overflow
    z_clipped = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z_clipped))


def logistic_prediction(w: np.array, x: np.array) -> float:
    """
    Making predictions based on the output of the logistic function
    """
    prob = logistic_function(w, x)
    return 1 if prob >= 0.5 else 0


def initialize_random_weights(p: int) -> np.array:
    """
    Generate a pseudorandom weight vector of dimension p.
    """
    return np.random.randn(p) * 0.01


def logistic_loss(w: np.array, x: np.array, c: int) -> float:
    """
    Calculate the logistic loss function
    """
    h = logistic_function(w, x)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    return -c * np.log(h) - (1 - c) * np.log(1 - h)


def train_logistic_regression_with_bgd(xs: np.array, cs: np.array, eta: float=1e-8, iterations: int=2000, validation_fraction: float=0) -> Tuple[np.array, List[float], List[float], List[float]]:
    """
    Fit a logistic regression model using the Batch Gradient Descent algorithm and
    return the learned weights as a numpy array.

    Arguments:
    - `xs`: feature vectors in the training dataset as a two-dimensional numpy array with shape (n, p+1)
    - `cs`: class values c(x) for every element in `xs` as a one-dimensional numpy array with length n
    - `eta`: the learning rate as a float value
    - `iterations': the number of iterations to run the algorithm for
    - 'validation_fraction': fraction of xs and cs used for validation (not for training)

    Returns:
    - the learned weights as a column vector, i.e. a two-dimensional numpy array with shape (1, p)
    - logistic loss value
    - misclassification rate of predictions on training part of xs/cs
    - misclassification rate of predictions on validation part of xs/cs
    """
    n = len(xs)
    p = xs.shape[1]
    
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
    
    # Initialize weights
    w = initialize_random_weights(p)
    
    loss_history = []
    train_misclass_history = []
    val_misclass_history = []
    
    for iteration in range(iterations):
        # Compute gradient using batch gradient descent
        gradient = np.zeros(p)
        total_loss = 0
        
        for i in range(len(xs_train)):
            x = xs_train[i]
            c = cs_train[i]
            h = logistic_function(w, x)
            gradient += (h - c) * x
            total_loss += logistic_loss(w, x, c)
        
        # Update weights
        w = w - eta * gradient
        
        # Calculate average loss
        avg_loss = total_loss / len(xs_train)
        loss_history.append(avg_loss)
        
        # Calculate misclassification rates
        train_predictions = np.array([logistic_prediction(w, x) for x in xs_train])
        train_misclass = misclassification_rate(cs_train, train_predictions)
        train_misclass_history.append(train_misclass)
        
        if validation_fraction > 0:
            val_predictions = np.array([logistic_prediction(w, x) for x in xs_val])
            val_misclass = misclassification_rate(cs_val, val_predictions)
            val_misclass_history.append(val_misclass)
        else:
            val_misclass_history.append(float('nan'))
    
    return w, loss_history, train_misclass_history, val_misclass_history


def plot_loss_and_misclassification_rates(loss: List[float],
                                          train_misclassification_rates: List[float],
                                          validation_misclassification_rates: List[float]):
    """
    Plots the normalized loss (divided by max(loss)) and both misclassification rates
    for each iteration.
    """
    iterations = range(len(loss))
    
    # Normalize loss
    max_loss = max(loss)
    normalized_loss = [l / max_loss for l in loss]
    
    plt.figure(figsize=(12, 5))
    
    # Plot normalized loss
    plt.subplot(1, 2, 1)
    plt.plot(iterations, normalized_loss, label='Normalized Loss', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Normalized Loss')
    plt.title('Training Loss Over Iterations')
    plt.legend()
    plt.grid(True)
    
    # Plot misclassification rates
    plt.subplot(1, 2, 2)
    plt.plot(iterations, train_misclassification_rates, label='Training Misclassification', color='green')
    if not np.isnan(validation_misclassification_rates[0]):
        plt.plot(iterations, validation_misclassification_rates, label='Validation Misclassification', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Misclassification Rate')
    plt.title('Misclassification Rates Over Iterations')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('loss_and_misclassification.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as loss_and_misclassification.png")

########################################################################
# Tests
import os
from pytest import approx


def test_logistic_function():
    x = np.array([1, 1, 2])
    assert logistic_function(np.array([0, 0, 0]), x) == approx(0.5)
    assert logistic_function(np.array([1e2, 1e2, 1e2]), x) == approx(1)
    assert logistic_function(np.array([-1e2, -1e2, -1e2]), x) == approx(0)
    assert logistic_function(np.array([1e2, -1e2, 0]), x) == approx(0.5)


def test_bgd():
    xs = np.array([
        [1, -1],
        [1, 2],
        [1, -2],
    ])
    cs = np.array([0, 1, 0])

    w, _, _, _ = train_logistic_regression_with_bgd(xs, cs, 0.1, 100)
    assert w @ [1, -1] < 0 and w @ [1, 2] > 0
    w, _, _, _ = train_logistic_regression_with_bgd(-xs, cs, 0.1, 100)
    assert w @ [1, -1] > 0 and w @ [1, 2] < 0



########################################################################
# Main program for running against the training dataset

if __name__ == "__main__":
    import pandas as pd
    import pytest
    import sys

    train_features_file_name = sys.argv[1]
    train_classes_file_name = sys.argv[2]
    test_features_file_name = sys.argv[3]
    test_predictions_file_name = sys.argv[4]

    print("(a)")
    xs = load_feature_vectors(train_features_file_name)
    xs_test = load_feature_vectors(test_features_file_name)
    cs = load_class_values(train_classes_file_name)
    # Print number of examples with each class
    n_human = np.sum(cs == 1)
    n_ai = np.sum(cs == 0)
    print(f"Number of human-written examples (class 1): {n_human}")
    print(f"Number of AI-generated examples (class 0): {n_ai}")
    print(f"Total examples: {len(cs)}")

    print("(b)")
    # Random classifier - randomly assign 0 or 1
    random_predictions = np.random.randint(0, 2, len(cs))
    random_misclass = misclassification_rate(cs, random_predictions)
    print(f"Misclassification rate of random classifier: {random_misclass:.4f}")

    print("(c)")
    test_c_result = pytest.main(['-k', 'test_logistic_function', '--tb=short', __file__])
    if test_c_result != 0:
        sys.exit(test_c_result)
    print("Test logistic function successful")

    print("(d)")
    test_d_result = pytest.main(['-k', 'test_bgd', '--tb=short', __file__])
    if test_d_result != 0:
        sys.exit(test_d_result)
    print("Test bgd successful")
    w, loss, train_misclassification_rates, validation_misclassification_rates = train_logistic_regression_with_bgd(xs, cs, validation_fraction = 0.2)

    print("(e)")
    plot_loss_and_misclassification_rates(loss, train_misclassification_rates, validation_misclassification_rates)

    print("(f)")
    # Predict on test set
    test_predictions = np.array([logistic_prediction(w, x) for x in xs_test])
    # Convert 0/1 to False/True
    test_predictions_bool = [True if p == 1 else False for p in test_predictions]
    
    # Write to file
    test_df = pd.DataFrame({
        'id': range(len(test_predictions_bool)),
        'is_human': test_predictions_bool
    })
    test_df.to_csv(test_predictions_file_name, sep='\t', index=False)
    print(f"Test predictions written to {test_predictions_file_name}")
    print(f"Number of test examples classified as human: {np.sum(test_predictions == 1)}")
    print(f"Number of test examples classified as AI: {np.sum(test_predictions == 0)}")
