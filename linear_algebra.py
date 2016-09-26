# -*- coding: iso-8859-15 -*-

from __future__ import division # want 3 / 2 == 1.5
import re, math, random # regexes, math functions, random numbers
import matplotlib.pyplot as plt # pyplot
from collections import defaultdict, Counter
from functools import partial

#  Functions for working with vectors

def vector_add(v, w):
    """
    Adds two vectors componentwise
    """
    return [v_i + w_i for v_i, w_i in zip(v,w)]

def vector_subtract(v, w):
    """
    Subtracts two vectors in componentwise
    """
    return [v_i - w_i for v_i, w_i in zip(v,w)]

def vector_sum(vectors):
    """
    Sum all corresponding elements.
    Start with first vector, loop over others, add to result
    """
    # result = vectors[0]
    # for vector in vectors[1:]:
    #     result = vector_add(result, vector)
    # return result

    return reduce(vector_add, vectors)

    # vector_sum = partial(reduce, vector_add)

def scalar_multiply(c, v):
    """
    c is a number, v is a vector
    """
    return [c * v_i for v_i in v]

def vector_mean(vectors):
    """
    Compute the vector where i-th element is the mean of the i-th elements
    of the input vectors
    """

def dot(v, w):
    """
    The DOT PRODUCT - Sum of component wise products
    v_1 * w_1 + ... + v_n * w_n
    """
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v):
    """
    Square a vector
    v_1 * v_1 + ... + v_n * v_n
    """
    return dot(v, v)

def magnitude(v):
    """
    Return the magnitude of the vector (sqrt of sum of squares)
    """
    return math.sqrt(sum_of_squares(v))

def squared_distance(v, w):
    """
    Compute distance between two vectors
    ((v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2)
    """
    return sum_of_squares(vector_subtract(v, w))

def distance(v, w):
    """
    Returns distance from squared_distance function
    """
    return math.sqrt(squared_distance(v, w))

#  Functions for working with matrices

def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols

def get_row(A, i):
    return A[i]             # A[i] is already the i-th row

def get_column(A, j):
    return [A_i[j] for A_i in A]      # jth element in A_i, for each row A_i

def make_matrix(num_rows, num_cols, entry_fn):
    """
    Reuturns num_rows x num_cols matrix with (i,j)th entry = entry_fn(i,j)
    """
    return [[entry_fn(i,j)                  # given i, create list
                for j in range(num_cols)]   # [entry_fn(i, 0), ...]
                for i in range(num_rows)]   # create line list for each i

def is_diagonal(i, j):
    """
    1's on the diagonal, 0's everywhere else!!!
    """
    return 1 if i == j else 0

# identity_matrix = make_matrix(5, 5, is_diagonal)

friendships = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0], # user 0
               [1, 0, 1, 1, 0, 0, 0, 0, 0, 0], # user 1
               [1, 1, 0, 1, 0, 0, 0, 0, 0, 0], # user 2
               [0, 1, 1, 0, 1, 0, 0, 0, 0, 0], # user 3
               [0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # user 4
               [0, 0, 0, 0, 1, 0, 1, 1, 0, 0], # user 5
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # user 6
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # user 7
               [0, 0, 0, 0, 0, 0, 1, 1, 0, 1], # user 8
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] # user 9


def matrix_add(A, B):
    """
    Add matrices if they are of same shape
    """
    if shape(A) != shape(B):
        raise ArithmeticError("Cannot add matrices of different shapes")
    num_rows, num_cols = shape(A)
    def entry_fn(i, j): return A[i][j] + B[i][j]

    return make_matrix(num_rows, num_cols, entry_fn)


def make_graph_dot_product_as_vector_projection(plt):
    v = [2, 1]
    w = [math.sqrt(.25), math.sqrt(.75)]
    c = dot(v, w)
    vonw = scalar_multiply(c, w)
    o = [0,0]

    plt.arrow(0, 0, v[0], v[1],
              width=0.002, head_width=.1, length_includes_head=True)
    plt.annotate("v", v, xytext=[v[0] + 0.1, v[1]])
    plt.arrow(0 ,0, w[0], w[1],
              width=0.002, head_width=.1, length_includes_head=True)
    plt.annotate("w", w, xytext=[w[0] - 0.1, w[1]])
    plt.arrow(0, 0, vonw[0], vonw[1], length_includes_head=True)
    plt.annotate(u"(v•w)w", vonw, xytext=[vonw[0] - 0.1, vonw[1] + 0.1])
    plt.arrow(v[0], v[1], vonw[0] - v[0], vonw[1] - v[1],
              linestyle='dotted', length_includes_head=True)
    plt.scatter(*zip(v,w,o),marker='.')
    plt.axis('equal')
    plt.show()
