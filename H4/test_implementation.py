#!/usr/bin/env python3
import numpy as np
from programming_exercise_neural_networks import *

# Test with the XOR example from the hint
np.random.seed(1)

xs = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])
cs = np.array([0, 1, 1, 0])

print("Testing with XOR dataset:")
print(f"xs shape: {xs.shape}")
print(f"cs shape: {cs.shape}")

# Test encode_class_values
cs_encoded = encode_class_values(cs, k=2)
print(f"\nEncoded classes:\n{cs_encoded}")

# Test train_multilayer_perceptron
print("\nTraining on XOR...")
Wh, Wo, train_hist, val_hist, weights_hist = train_multilayer_perceptron(
    xs, cs, l=3, eta=0.1, iterations=100, validation_fraction=0
)

print(f"\nFinal Wh:\n{Wh.round(2)}")
print(f"\nFinal Wo:\n{Wo.round(2)}")

# Test predictions
probs = predict_probabilities(Wh, Wo, xs)
preds = predict(Wh, Wo, xs)

print(f"\nPredictions: {preds}")
print(f"Truth:       {cs}")
print(f"Misclassification rate: {misclassification_rate(cs, preds):.4f}")
