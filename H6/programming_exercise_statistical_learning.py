#!/usr/bin/env python3
import numpy as np
import pandas as pd
from math import prod
############################################################################
# Starter code for exercise: Naive Bayes for Generative Authorship Detection
############################################################################

import re
from collections import Counter

# Global vocab for text features (built on training, reused for val/test)
_VOCAB = None
_VOCAB_SIZE = 2000

_STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","for","to","of","in","on","at","by","with","from",
    "is","are","was","were","be","been","being","it","this","that","these","those","as","not","no","do","does",
    "did","so","such","can","could","will","would","should","may","might","you","your","we","our","they","their",
    "i","me","my","he","him","his","she","her","its","them","what","which","who","whom","when","where","why","how"
}

_WORD_RE  = re.compile(r"[a-z]{2,}")
_URL_RE   = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")

def _tokenize(text: str):
    text = text.lower()
    text = _URL_RE.sub(" __url__ ", text)
    text = _EMAIL_RE.sub(" __email__ ", text)
    text = re.sub(r"\d+", " __num__ ", text)
    tokens = _WORD_RE.findall(text)
    return [t for t in tokens if t not in _STOPWORDS]

GROUP = "02" # TODO: write in your group name here


def load_bow_feature_vectors(filename: str) -> np.array:
    """
    Load the Bag-of-words feature vectors from the given file and return
    them as a two-dimensional numpy array with shape (n, p), where n is the number
    of examples in the dataset and p is the number of features per example.
    """
    return np.load(filename)

# From last exercise sheet
def load_class_values(filename: str) -> np.array:
    """
    Load the class values from the dataset in the given file and return
    them as a one-dimensional numpy array.
    """
    return np.ravel((pd.read_csv(filename, sep='\t', usecols=["is_human"]).to_numpy() > 0) * 1) 


def misclassification_rate(cs: np.array, ys: np.array) -> float:
    """
    This function takes two vectors with gold and predicted labels and
    returns the percentage of positions where truth and prediction disagree
    """
    if len(cs) == 0:
        return float('nan')
    else:
        hits = np.sum(cs == ys)
        return 1 - hits / len(cs)
    


def class_priors(cs: np.ndarray) -> dict:
    """Compute the prior probabilities P(C=c) for all the distinct classes c in the given dataset.

    Args:
        cs (np.ndarray): one-dimensional array of values c(x) for all examples x from the dataset D

    Returns:
        dict: a dictionary mapping each distinct class to its prior probability
    """
    # TODO: your code here
    values, counts = np.unique(cs, return_counts=True)
    n = len(cs)
    return {v: c / n for v, c in zip(values, counts)}

def extract_features(filename: str) -> np.array:
    """
    Load the TSV file from the given filename and return a numpy array with the
    features extracted from the text.
    """
    data = pd.read_csv(filename, sep='\t')
    # TODO: your code here 
    global _VOCAB

    # guard against your exact error (passing .npy here)
    if filename.endswith(".npy"):
        raise ValueError(f"extract_features expects a .tsv text file, got: {filename}")
    texts = data["text"].astype(str).tolist()

    # Build vocab only once (should happen on training texts)
    if _VOCAB is None:
        df = Counter()
        for t in texts:
            for w in set(_tokenize(t)):  # document frequency
                df[w] += 1
        most_common = [w for w, _ in df.most_common(_VOCAB_SIZE)]
        _VOCAB = {w: i for i, w in enumerate(most_common)}

    n = len(texts)
    p_words = len(_VOCAB)
    p_extra = 6
    X = np.zeros((n, p_words + p_extra), dtype=np.int16)

    for i, raw in enumerate(texts):
        toks = set(_tokenize(raw))

        # binary word presence
        for w in toks:
            j = _VOCAB.get(w)
            if j is not None:
                X[i, j] = 1

        # extra discrete features (small buckets)
        has_url = 1 if _URL_RE.search(raw) else 0
        has_at  = 1 if "@" in raw else 0

        num_excl = raw.count("!")
        num_q    = raw.count("?")

        digits = len(re.findall(r"\d", raw))
        digits_b = 0 if digits == 0 else (1 if digits == 1 else 2)

        tok_len = len(_tokenize(raw))
        len_b = 0 if tok_len < 80 else (1 if tok_len <= 200 else 2)

        base = p_words
        X[i, base + 0] = has_url
        X[i, base + 1] = has_at
        X[i, base + 2] = 0 if num_excl == 0 else (1 if num_excl == 1 else 2)
        X[i, base + 3] = 0 if num_q == 0 else (1 if num_q == 1 else 2)
        X[i, base + 4] = digits_b
        X[i, base + 5] = len_b

    return X

def conditional_probabilities(xs: np.ndarray, cs: np.ndarray) -> dict:
    """Compute the conditional probabilities P(B_j = x_j | C = c) for all combinations of feature B_j, feature value x_j and class c found in the given dataset.

    Args:
        xs (np.ndarray): n-by-p array with n points of p attributes each
        cs (np.ndarray): one-dimensional n-element array with values c(x)

    Returns:
        dict: nested dictionary d with d[c][B_j][x_j] = P(B_j = x_j | C=c)
    """
    # TODO: your code here
    # nested dict: p[c][j][xj] = P(B_j=xj | C=c)
    out = {}
    classes = np.unique(cs)
    p = xs.shape[1]

    for c in classes:
        Xc = xs[cs == c]
        n_c = len(Xc)
        out[c] = {}
        for j in range(p):
            col = Xc[:, j]
            vals, cnts = np.unique(col, return_counts=True)
            out[c][j] = {v: cnt / n_c for v, cnt in zip(vals, cnts)}
    return out


class NaiveBayesClassifier:
    def fit(self, xs: np.ndarray, cs: np.ndarray):
        """Fit a Naive Bayes model on the given dataset

        Args:
            xs (np.ndarray): n-by-p array of feature vectors
            cs (np.ndarray): n-element array of class values
        """
        # TODO: your code here
        self.classes_ = list(np.unique(cs))
        self.priors_ = class_priors(cs)
        self.cond_ = conditional_probabilities(xs, cs)
    
    def predict(self, x: np.ndarray) -> str:
        """Generate a prediction for the data point x

        Args:
            x (np.ndarray): a p-dimensional feature vector

        Returns:
            str: the most probable class for x
        """
        # TODO: your code here
        best_c = None
        best_logp = -np.inf

        for c in self.classes_:
            # log prior
            logp = np.log(self.priors_[c])

            # add log likelihoods for seen feature-values
            for j, xj in enumerate(x):
                pj_dict = self.cond_[c].get(j, {})
                if xj in pj_dict:
                    logp += np.log(pj_dict[xj])
                else:
                    # unseen value -> ignore feature (no evidence)
                    logp += 0.0

            if logp > best_logp:
                best_logp = logp
                best_c = c

        return best_c
        


def train_and_predict(training_features_file_name: str, training_labels_file_name: str, 
                      validation_features_file_name: str, validation_labels_file_name: str,
                      test_features_file_name: str) -> np.ndarray:
    """Train a model on the given training dataset, and predict the class values
    for the given testing dataset. Report the misclassification rate on the training
    and validation sets.

    Return an array with the predicted class values, in the same order as the
    examples in the testing dataset.
    """
    # TODO: Your code here
    Xtr = load_bow_feature_vectors(training_features_file_name)
    ctr = load_class_values(training_labels_file_name)

    Xva = load_bow_feature_vectors(validation_features_file_name)
    cva = load_class_values(validation_labels_file_name)

    Xte = load_bow_feature_vectors(test_features_file_name)

    clf = NaiveBayesClassifier()
    clf.fit(Xtr, ctr)

    # predict train/val
    ytr = np.array([clf.predict(x) for x in Xtr])
    yva = np.array([clf.predict(x) for x in Xva])

    print("Train misclassification rate:", misclassification_rate(ctr, ytr))
    print("Val misclassification rate:", misclassification_rate(cva, yva))

    # predict test
    yte = np.array([clf.predict(x) for x in Xte])
    return yte

def train_and_predict_from_texts(train_texts_tsv, train_labels_tsv,
                                 val_texts_tsv, val_labels_tsv,
                                 test_texts_tsv):
    global _VOCAB
    _VOCAB = None  # reset vocab so training builds it

    Xtr = extract_features(train_texts_tsv)
    ctr = load_class_values(train_labels_tsv)

    Xva = extract_features(val_texts_tsv)
    cva = load_class_values(val_labels_tsv)

    Xte = extract_features(test_texts_tsv)

    clf = NaiveBayesClassifier()
    clf.fit(Xtr, ctr)

    ytr = np.array([clf.predict(x) for x in Xtr])
    yva = np.array([clf.predict(x) for x in Xva])

    print("Train misclassification rate:", misclassification_rate(ctr, ytr))
    print("Val misclassification rate:", misclassification_rate(cva, yva))

    return np.array([clf.predict(x) for x in Xte])

########################################################################
# Tests
import os
from pytest import approx

train_features_file_name = os.path.join(os.path.dirname(__file__), 'data/bow-features-train.npy')
train_classes_file_name = os.path.join(os.path.dirname(__file__), 'data/labels-train.tsv')
val_features_file_name = os.path.join(os.path.dirname(__file__), 'data/bow-features-val.npy')
val_classes_file_name = os.path.join(os.path.dirname(__file__), 'data/labels-val.tsv')
test_features_file_name = os.path.join(os.path.dirname(__file__), 'data/bow-features-test.npy')
train_texts_file_name = os.path.join(os.path.dirname(__file__), 'data/texts-training.tsv')
val_texts_file_name   = os.path.join(os.path.dirname(__file__), 'data/texts-val.tsv')
test_texts_file_name  = os.path.join(os.path.dirname(__file__), 'data/texts-test.tsv')

def test_that_the_group_name_is_there():
    import re
    assert re.match(r'^[0-9]{1,3}$', GROUP), \
        "Please write your group name in the variable at the top of the file!"

def test_that_training_features_are_here():
    assert os.path.isfile(train_features_file_name), \
        "Please put the training dataset file next to this script!"

def test_that_training_classes_are_here():
    assert os.path.isfile(train_classes_file_name), \
        "Please put the validation dataset file next to this script!"
    
def test_that_validation_features_are_here():
    assert os.path.isfile(val_features_file_name), \
        "Please put the validation dataset file next to this script!"

def test_that_validation_classes_are_here():
    assert os.path.isfile(val_classes_file_name), \
        "Please put the validation dataset file next to this script!"

def test_that_test_features_are_here():
    assert os.path.isfile(test_features_file_name), \
        "Please put the test dataset file next to this script!"

def test_class_priors():
    cs = np.array(list('abcababa'))
    priors = class_priors(cs)
    assert priors == dict(a=0.5, b=0.375, c=0.125)

def test_conditional_probabilities():
    cs = np.array(list('aabb'))
    xs = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [2, 0, 0],
        [2, 1, 0]
    ])

    p = conditional_probabilities(xs, cs)

    assert p['a'][0][1] == 0.5
    assert p['a'][0][0] == 0.5
    assert p['b'][0][2] == 1
    assert p['a'][1][0] == 0.5
    assert p['a'][1][1] == 0.5
    assert p['b'][1][0] == 0.5
    assert p['b'][1][1] == 0.5
    assert p['a'][2][1] == 0.5
    assert p['a'][2][0] == 0.5
    assert p['b'][2][0] == 1

### example dataset from the lecture
xs_example = np.array([x.split() for x in """sunny hot high weak
sunny hot high strong
overcast hot high weak
rain mild high weak
rain cold normal weak
rain cold normal strong
overcast cold normal strong
sunny mild high weak
sunny cold normal weak
rain mild normal weak
sunny mild normal strong
overcast mild high strong
overcast hot normal weak
rain mild high strong""".split('\n')])

cs_example = np.array("no no yes yes yes no yes no yes yes yes yes yes no".split())

def test_classifier():
    clf = NaiveBayesClassifier()
    clf.fit(xs_example, cs_example)
    pred = clf.predict(np.array('sunny cold high strong'.split()))
    assert pred == 'no', 'should classify example from the lecture correctly'

def test_classifier_unknown_value():
    clf = NaiveBayesClassifier()
    clf.fit(xs_example, cs_example)
    pred = clf.predict(np.array('sunny hot dry none'.split()))
    assert pred == 'no', 'should handle unknown feature values'


########################################################################
# Main program for running against the training dataset

if __name__ == "__main__":
    import pandas as pd
    import pytest
    import sys
    test_result = pytest.main(['--tb=short', __file__])
    if test_result != 0:
        sys.exit(test_result)
    print("Great! All tests passed!")
    print("Running train_and_predict.")
    preds = train_and_predict_from_texts(train_texts_file_name, train_classes_file_name,
                              val_texts_file_name, val_classes_file_name,
                              test_texts_file_name)
    if preds is not None:
        print("Saving predictions.")
        pd.DataFrame(preds).to_csv(f"naive-bayes-predictions-test-group-{GROUP}.tsv", header=False, index=False, sep='\t')