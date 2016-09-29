from __future__ import division
from collections import Counter, defaultdict
from functools import partial
import math, random


def entropy(class_probabilities):
    """
    Compute entropy for list of class probabilities
    """
    return sum(-p * math.log(p, 2) for p in class_probabilities if p)


def class_probabilities(labels):
    """
    Returns probabilities for each class based on labels provided
    """
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]


def data_entropy(labeled_data):
    """
    Returns entropy for data
    """
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)


def partition_entropy(subsets):
    """
    Find entropy for partition of data in subsets
    """
    total_count = sum(len(subset) for subset in subsets)

    return sum(data_entropy(subset) * len(subset) / total_count
               for subset in subsets)


def group_by(items, key_fn):
    """
    Returns defaultdict(list), each input item is in the list whose key is in
    key_fn(item)
    """
    groups = defaultdict(list)
    for item in items:
        key = key_fn(item)
        groups[key].append(item)
    return groups


def partition_by(inputs, attribute):
    """
    Each input is a pair (attribute_dict, label).
    Returns dict : attribute_value --> inputs
    """
    return group_by(inputs, lambda x: x[0][attribute])


def partition_entropy_by(inputs, attribute):
    """
    Computes entropy corresponding to given partition
    """
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())


def classify(tree, input):
    """
    Classify input using given decision tree
    """
    #  If this is leave node, return its value
    if tree in [True, False]:
        return tree

    #  Otherwise find correct subtree
    attribute, subtree_dict = tree

    subtree_key = input.get(attribute)      #  None if input is missing attribute

    if subtree_key not in subtree_dict:     #  If no subtree for key
        subtree_key = None                  #  Use None subtree

    subtree = subtree_dict[subtee_key]      #  Choose appropriate subtree
    return classify(subtree, input)         #  Use to classify input



def build_tree_id3(inputs, split_candidates=None):
    """
    Built tree from training data
    """
    #  If first pass, all keys of first input are split candidates
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()

    #  Count Trues and Falses in inputs
    num_inputs = len(inputs)
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs - num_trues

    if num_trues == 0:          #  If only Falses left
        return False            #  Return False leaf
    if num_falses == 0:         #  If only Trues left
        return Trues            #  Return True leaf
    if not split_candidates:    #  if no split_candidates left
        return num_trees >= num_falses  #  Return majority leaf

    #  Otherwise, split on best attribute
    best_attribute = min(split_candidates,
                         key=partial(partition_entropy_by,
                         inputs))

    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates if a != best_attribute]

    #  Recursively build subtrees
    subtrees = {attribute : build_tree_id3(subset, new_candidates)
                for attribute, subset in partitions.iteritems()}

    subtrees[None] = num_trees > num_falses     #  Default case

    return (best_attribute, subtrees)


def forest_classify(trees, input):
    """
    Random forest, vote on classification
    """
    votes = [classify(tree, input) for tree in trees]
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]


if __name__ == "__main__":

    inputs = [
        ({'level':'Senior','lang':'Java','tweets':'no','phd':'no'},   False),
        ({'level':'Senior','lang':'Java','tweets':'no','phd':'yes'},  False),
        ({'level':'Mid','lang':'Python','tweets':'no','phd':'no'},     True),
        ({'level':'Junior','lang':'Python','tweets':'no','phd':'no'},  True),
        ({'level':'Junior','lang':'R','tweets':'yes','phd':'no'},      True),
        ({'level':'Junior','lang':'R','tweets':'yes','phd':'yes'},    False),
        ({'level':'Mid','lang':'R','tweets':'yes','phd':'yes'},        True),
        ({'level':'Senior','lang':'Python','tweets':'no','phd':'no'}, False),
        ({'level':'Senior','lang':'R','tweets':'yes','phd':'no'},      True),
        ({'level':'Junior','lang':'Python','tweets':'yes','phd':'no'}, True),
        ({'level':'Senior','lang':'Python','tweets':'yes','phd':'yes'},True),
        ({'level':'Mid','lang':'Python','tweets':'no','phd':'yes'},    True),
        ({'level':'Mid','lang':'Java','tweets':'yes','phd':'no'},      True),
        ({'level':'Junior','lang':'Python','tweets':'no','phd':'yes'},False)
    ]

    for key in ['level', 'lang', 'tweets', 'phd']:
        print key, partition_entropy_by(inputs, key)
    print

    senior_inputs = [(input, label)
                      for input, label in inputs if input["level"] == "Senior"]

    for key in ['lang', 'tweets', 'phd']:
        print key, partition_entropy_by(senior_inputs, key)
    print

    print "BUILDING THE TREE"
    tree = build_tree_id3(inputs)
    print tree

    print "Junior / Java / tweets / no phd", classify(tree,
        { "level" : "Junior",
          "lang" : "Java",
          "tweets" : "yes",
          "phd" : "no"} )

    print "Junior / Java / tweets / phd", classify(tree,
        { "level" : "Junior",
                 "lang" : "Java",
                 "tweets" : "yes",
                 "phd" : "yes"} )

    print "Intern", classify(tree, { "level" : "Intern" } )
    print "Senior", classify(tree, { "level" : "Senior" } )
