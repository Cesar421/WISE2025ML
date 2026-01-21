"""
Exercise 7: Argument Quality Prediction with CART Decision Trees
Implements the CART algorithm for decision tree classification.
"""

import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Any


class TreeNode:
    """
    Represents a node in the decision tree.
    """
    def __init__(self, 
                 feature_index: Optional[int] = None,
                 threshold: Optional[float] = None,
                 left: Optional['TreeNode'] = None,
                 right: Optional['TreeNode'] = None,
                 value: Optional[Any] = None,
                 is_leaf: bool = False):
        """
        Initialize a tree node.
        
        Args:
            feature_index: Index of the feature to split on
            threshold: Threshold value for the split
            left: Left child node (examples <= threshold)
            right: Right child node (examples > threshold)
            value: Class label (for leaf nodes)
            is_leaf: Whether this is a leaf node
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf


def load_data(features_file: str, labels_file: str = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load feature and label data from TSV files.
    
    Args:
        features_file: Path to features TSV file
        labels_file: Path to labels TSV file (optional)
    
    Returns:
        Tuple of (features array, labels array or None)
    """
    # Load features - exclude the 'id' column
    features_df = pd.read_csv(features_file, sep='\t')
    if 'id' in features_df.columns:
        features_df = features_df.drop('id', axis=1)
    X = features_df.values
    
    # Load labels if provided - only load the label column, not 'id'
    y = None
    if labels_file:
        labels_df = pd.read_csv(labels_file, sep='\t')
        # Get the label column (second column, assuming first is 'id')
        label_col = labels_df.columns[-1]  # Get the last column (label column)
        y = labels_df[label_col].values
        # Convert boolean to int if needed
        if y.dtype == bool:
            y = y.astype(int)
    
    return X, y


def most_common_class(C: np.ndarray) -> Any:
    """
    (a) Find the most common class in the dataset.
    
    Args:
        C: Array of class labels
    
    Returns:
        The most common class label
    """
    unique, counts = np.unique(C, return_counts=True)
    return unique[np.argmax(counts)]


def gini_impurity(C: np.ndarray) -> float:
    """
    (b) Compute the Gini index for the given set of example classes C.
    
    Gini impurity = 1 - sum(p_i^2) for all classes i
    where p_i is the probability of class i
    
    Args:
        C: Array of class labels
    
    Returns:
        Gini impurity value
    """
    if len(C) == 0:
        return 0.0
    
    # Count occurrences of each class
    unique, counts = np.unique(C, return_counts=True)
    
    # Calculate probabilities
    probabilities = counts / len(C)
    
    # Gini impurity = 1 - sum(p_i^2)
    gini = 1.0 - np.sum(probabilities ** 2)
    
    return gini


def gini_impurity_reduction(C: np.ndarray, C_left: np.ndarray, C_right: np.ndarray) -> float:
    """
    (c) Compute the Gini impurity reduction of a binary split.
    
    Reduction = Gini(C) - (|C_left|/|C| * Gini(C_left) + |C_right|/|C| * Gini(C_right))
    
    Args:
        C: Array of all class labels before split
        C_left: Array of class labels in left split
        C_right: Array of class labels in right split
    
    Returns:
        Gini impurity reduction value
    """
    n = len(C)
    if n == 0:
        return 0.0
    
    n_left = len(C_left)
    n_right = len(C_right)
    
    # Weighted Gini impurity after split
    weighted_gini = (n_left / n) * gini_impurity(C_left) + (n_right / n) * gini_impurity(C_right)
    
    # Gini impurity reduction
    reduction = gini_impurity(C) - weighted_gini
    
    return reduction


def possible_thresholds(X: np.ndarray, feature_index: int) -> List[float]:
    """
    (d) Return all possible thresholds for splitting the example set X along the given feature.
    Pick thresholds as the mid-point between all pairs of distinct, consecutive values in ascending order.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        feature_index: Index of the feature to consider
    
    Returns:
        List of possible threshold values
    """
    # Get all values for this feature
    feature_values = X[:, feature_index]
    
    # Get unique values in ascending order
    unique_values = np.unique(feature_values)
    
    # If there's only one unique value, no split is possible
    if len(unique_values) <= 1:
        return []
    
    # Calculate mid-points between consecutive values
    thresholds = []
    for i in range(len(unique_values) - 1):
        threshold = (unique_values[i] + unique_values[i + 1]) / 2.0
        thresholds.append(threshold)
    
    return thresholds


def find_best_split(X: np.ndarray, C: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
    """
    (e) Find the best split based on the Gini impurity reduction for the given set of examples X
    and the given set of classes C.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        C: Array of class labels
    
    Returns:
        Tuple of (best_feature_index, best_threshold, best_reduction)
        Returns (None, None, 0.0) if no valid split is found
    """
    best_feature = None
    best_threshold = None
    best_reduction = 0.0
    
    n_features = X.shape[1]
    
    # Try each feature
    for feature_index in range(n_features):
        # Get all possible thresholds for this feature
        thresholds = possible_thresholds(X, feature_index)
        
        # Try each threshold
        for threshold in thresholds:
            # Split the data
            left_mask = X[:, feature_index] <= threshold
            right_mask = ~left_mask
            
            C_left = C[left_mask]
            C_right = C[right_mask]
            
            # Skip if split results in empty subset
            if len(C_left) == 0 or len(C_right) == 0:
                continue
            
            # Calculate Gini impurity reduction
            reduction = gini_impurity_reduction(C, C_left, C_right)
            
            # Update best split if this is better
            if reduction > best_reduction:
                best_reduction = reduction
                best_feature = feature_index
                best_threshold = threshold
    
    return best_feature, best_threshold, best_reduction


def id3_cart(X: np.ndarray, C: np.ndarray, depth: int = 0, max_depth: int = None, 
             min_samples_split: int = 2, min_impurity_decrease: float = 0.0) -> TreeNode:
    """
    (f) Construct a CART decision tree with the modified ID3 algorithm.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        C: Array of class labels
        depth: Current depth of the tree
        max_depth: Maximum depth of the tree (None for unlimited)
        min_samples_split: Minimum number of samples required to split
        min_impurity_decrease: Minimum impurity decrease required for split
    
    Returns:
        Root node of the decision tree
    """
    # Stopping criteria
    # 1. All examples have the same class
    if len(np.unique(C)) == 1:
        return TreeNode(value=C[0], is_leaf=True)
    
    # 2. Maximum depth reached
    if max_depth is not None and depth >= max_depth:
        return TreeNode(value=most_common_class(C), is_leaf=True)
    
    # 3. Not enough samples to split
    if len(C) < min_samples_split:
        return TreeNode(value=most_common_class(C), is_leaf=True)
    
    # Find the best split
    best_feature, best_threshold, best_reduction = find_best_split(X, C)
    
    # 4. No valid split found or impurity reduction too small
    if best_feature is None or best_reduction <= min_impurity_decrease:
        return TreeNode(value=most_common_class(C), is_leaf=True)
    
    # Split the data
    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask
    
    X_left, C_left = X[left_mask], C[left_mask]
    X_right, C_right = X[right_mask], C[right_mask]
    
    # Recursively build left and right subtrees
    left_child = id3_cart(X_left, C_left, depth + 1, max_depth, 
                          min_samples_split, min_impurity_decrease)
    right_child = id3_cart(X_right, C_right, depth + 1, max_depth, 
                           min_samples_split, min_impurity_decrease)
    
    # Create internal node
    return TreeNode(feature_index=best_feature, threshold=best_threshold,
                    left=left_child, right=right_child, is_leaf=False)


def predict_single(node: TreeNode, x: np.ndarray) -> Any:
    """
    Predict the class for a single example.
    
    Args:
        node: Current tree node
        x: Feature vector for a single example
    
    Returns:
        Predicted class label
    """
    if node.is_leaf:
        return node.value
    
    # Traverse tree based on feature value and threshold
    if x[node.feature_index] <= node.threshold:
        return predict_single(node.left, x)
    else:
        return predict_single(node.right, x)


def predict(tree: TreeNode, X: np.ndarray) -> np.ndarray:
    """
    Predict classes for multiple examples.
    
    Args:
        tree: Root node of the decision tree
        X: Feature matrix (n_samples, n_features)
    
    Returns:
        Array of predicted class labels
    """
    predictions = np.array([predict_single(tree, x) for x in X])
    return predictions


def calculate_misclassification_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the misclassification rate.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
    
    Returns:
        Misclassification rate (proportion of incorrect predictions)
    """
    return np.mean(y_true != y_pred)


def train_and_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                      max_depth: int = None, min_samples_split: int = 2,
                      min_impurity_decrease: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    (g) Train the model on the training set and return the predictions for the test set.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        X_test: Test feature matrix
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum number of samples required to split
        min_impurity_decrease: Minimum impurity decrease required for split
    
    Returns:
        Tuple of (test predictions, training misclassification rate)
    """
    # Train the CART decision tree
    tree = id3_cart(X_train, y_train, max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_impurity_decrease=min_impurity_decrease)
    
    # Make predictions on training set
    y_train_pred = predict(tree, X_train)
    
    # Calculate training misclassification rate
    train_misclassification_rate = calculate_misclassification_rate(y_train, y_train_pred)
    
    # Make predictions on test set
    y_test_pred = predict(tree, X_test)
    
    return y_test_pred, train_misclassification_rate


def save_predictions(predictions: np.ndarray, output_file: str):
    """
    Save predictions to a TSV file.
    
    Args:
        predictions: Array of predicted class labels
        output_file: Path to output file
    """
    df = pd.DataFrame({'prediction': predictions})
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Predictions saved to {output_file}")


def main():
    """
    Main function to run the decision tree algorithm.
    
    Usage:
        python3 programming_exercise_decision_trees.py 
            features-train-cleaned.tsv quality-scores-train-cleaned.tsv 
            features-test-cleaned.tsv
    """
    if len(sys.argv) != 4:
        print("Usage: python3 programming_exercise_decision_trees.py "
              "features-train.tsv labels-train.tsv features-test.tsv")
        sys.exit(1)
    
    features_train_file = sys.argv[1]
    labels_train_file = sys.argv[2]
    features_test_file = sys.argv[3]
    
    # Load data
    print("Loading data...")
    X_train, y_train = load_data(features_train_file, labels_train_file)
    X_test, _ = load_data(features_test_file)
    
    print(f"Training set: {X_train.shape[0]} examples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} examples")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Train model and make predictions
    print("\nTraining CART decision tree...")
    
    # Basic model without depth limit
    y_test_pred, train_misclassification = train_and_predict(
        X_train, y_train, X_test, max_depth=None
    )
    
    print(f"\nTraining misclassification rate (no depth limit): {train_misclassification:.4f}")
    
    # Try with depth limit to reduce overfitting
    print("\nTrying different max_depth values to find optimal...")
    best_depth = None
    best_score = float('inf')
    
    # Use a simple validation approach: split training data
    n_train = int(0.8 * len(X_train))
    X_train_split = X_train[:n_train]
    y_train_split = y_train[:n_train]
    X_val = X_train[n_train:]
    y_val = y_train[n_train:]
    
    for depth in range(1, 21):
        tree = id3_cart(X_train_split, y_train_split, max_depth=depth)
        y_val_pred = predict(tree, X_val)
        val_error = calculate_misclassification_rate(y_val, y_val_pred)
        
        if val_error < best_score:
            best_score = val_error
            best_depth = depth
        
        if depth <= 10 or depth % 5 == 0:
            print(f"  max_depth={depth}: validation error = {val_error:.4f}")
    
    print(f"\nBest max_depth found: {best_depth} (validation error: {best_score:.4f})")
    
    # Train final model with best depth on full training set
    print(f"\nTraining final model with max_depth={best_depth}...")
    y_test_pred_final, train_misclassification_final = train_and_predict(
        X_train, y_train, X_test, max_depth=best_depth
    )
    
    print(f"Final training misclassification rate: {train_misclassification_final:.4f}")
    
    # Save predictions
    output_file = "predictions-test.tsv"
    save_predictions(y_test_pred_final, output_file)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
