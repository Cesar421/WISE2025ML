# Exercise 7: Argument Quality Prediction with CART Decision Trees

## Solution Overview

This solution implements the CART (Classification and Regression Trees) algorithm for argument quality prediction. All required functions from the exercise have been implemented.

## Implemented Functions

### (a) `most_common_class(C)`
Finds the most common class in the dataset by counting occurrences and returning the class with maximum count.

### (b) `gini_impurity(C)`
Computes the Gini index using the formula: `Gini = 1 - Σ(p_i²)` where p_i is the probability of class i.

### (c) `gini_impurity_reduction(C, C_left, C_right)`
Computes the Gini impurity reduction of a binary split using weighted average:
`Reduction = Gini(C) - (|C_left|/|C| * Gini(C_left) + |C_right|/|C| * Gini(C_right))`

### (d) `possible_thresholds(X, feature_index)`
Returns all possible split thresholds for a feature by computing mid-points between consecutive distinct values in ascending order.

### (e) `find_best_split(X, C)`
Finds the best split (feature and threshold) that maximizes Gini impurity reduction across all features and thresholds.

### (f) `id3_cart(X, C, depth, max_depth, min_samples_split, min_impurity_decrease)`
Constructs a CART decision tree using the modified ID3 algorithm with the following stopping criteria:
- All examples have the same class
- Maximum depth reached
- Not enough samples to split
- No valid split found or impurity reduction too small

### (g) `train_and_predict(X_train, y_train, X_test, ...)`
Trains the model on the training set and returns predictions for the test set along with the training misclassification rate.

## Usage

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Program
```bash
python programming_exercise_decision_trees.py features-train-cleaned.tsv quality-scores-train-cleaned.tsv features-test-cleaned.tsv
```

### Expected Files
- `features-train-cleaned.tsv`: Training feature vectors
- `quality-scores-train-cleaned.tsv`: Training quality scores (labels)
- `features-test-cleaned.tsv`: Test feature vectors

### Output
- `predictions-test.tsv`: Predictions for the test set
- Console output showing:
  - Training misclassification rate
  - Validation results for different max_depth values
  - Best max_depth found
  - Final training misclassification rate

## Features

### Basic Implementation
- All required functions (a-g) are fully implemented
- Uses numpy for efficient numerical operations
- Follows CART algorithm as described in the slides

### Improvements Implemented
1. **Validation-based Depth Selection**: The program splits the training data into training and validation sets to find the optimal max_depth parameter.

2. **Stopping Criteria**: Multiple stopping criteria implemented:
   - `max_depth`: Maximum tree depth
   - `min_samples_split`: Minimum samples required to split a node
   - `min_impurity_decrease`: Minimum impurity decrease required for a split

3. **Overfitting Prevention**: 
   - Uses validation set to select optimal depth
   - Implements early stopping criteria
   - Prevents splits that don't improve model quality

## Training Misclassification Rate

The training misclassification rate is computed and displayed in the console output. Without depth limiting, the tree may overfit (very low or zero training error). The optimized version uses validation to find a good balance between training accuracy and generalization.

## Algorithm Details

### Tree Structure
Each node contains:
- `feature_index`: Feature to split on (internal nodes)
- `threshold`: Split threshold (internal nodes)
- `left`: Left child (examples ≤ threshold)
- `right`: Right child (examples > threshold)
- `value`: Class label (leaf nodes)
- `is_leaf`: Whether node is a leaf

### Prediction Process
For each example:
1. Start at root node
2. If leaf node, return its class value
3. Otherwise, compare feature value to threshold
4. Recursively traverse left (≤) or right (>) child
5. Return leaf node's class value

## Notes

- All features are assumed to be numeric (as specified in the exercise)
- The implementation follows the CART algorithm from slides ML:VI-109 and ML:VI-22
- Thresholds are computed as mid-points between consecutive distinct values
- The Gini impurity is used as the split criterion
