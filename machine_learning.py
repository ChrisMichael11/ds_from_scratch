from __future__ import division
from collections import Counter
import math, random

##  DATA splitting
def split_data(data, prob):
    """
    Split data into fractions, [prob, 1 - prob]
    """
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results


def train_test_split(x, y, test_pct):
    """
    Create train/test/split
    """
    data = zip(x,y)
    train, test = split_data(data, 1 - test_pct)
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    return x_train, x_test, y_train, y_test


## CHECKING FOR CORRECTNESS

def accuracy(tp, fp, fn, tn):
    """
    Accuracy - Fraction of correct predictions
    """
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total


def precision(tp, fp, fn, tn):
    """
    Precision - How accurate positive predictions were
    """
    return tp / (tp + fp)


def recall(tp, fp, fn, tn):
    """
    Recall - what fraction of positives the model identified
    """
    return tp / (tp + fn)


def f1_score(tp, fp, fn, tn):
    """
    F1_score - harmonic mean of precision and recall
    """
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)

    return 2 * p * r / (p + r)


if __name__ == "__main__":

    print "accuracy(70, 4930, 13930, 981070)", accuracy(70, 4930, 13930, 981070)
    print "precision(70, 4930, 13930, 981070)", precision(70, 4930, 13930, 981070)
    print "recall(70, 4930, 13930, 981070)", recall(70, 4930, 13930, 981070)
    print "f1_score(70, 4930, 13930, 981070)", f1_score(70, 4930, 13930, 981070)
