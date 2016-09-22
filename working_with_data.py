from __future__ import division
from collections import Counter, defaultdict
from functools import partial
from linear_algebra import shape, get_row, get_column, make_matrix, vector_mean, \
        vector_sum, dot, magnitude, vector_subtract, scalar_multiply
from statistics import correlation, standard_deviation, mean
from probability import inverse_normal_cdf
from gradient_descent import maximize_batch
import math, random, csv
import matplotlib.pyplot as plt
import dateutil.parser


def bucketize(point, bucket_size):
    """
    Floor the point to the next lower multiple of bucket_size
    """
    return bucket_size * math.floor(point / bucket_size)


def make_histogram(points, bucket_size):
    """
    Bucket points and counts how many in each bucket
    """
    return Counter(bucketize(point, bucket_size) for point in points)


def plot_histogram(points, bucket_size, title=""):
    historgram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)
    plt.show()


def compare_two_distributions():
    """
    Compare two distrubtions w/Histogram
    """
    random.seed(11)

    uniform = [random.randrange(-100, 101) for _ in range(200)]
    normal = [57 * inverse_normal_cdf(random.random()) for _ in range(200)]

    plot_histogram(uniform, 10, "Uniform Histogram")
    plot_histogram(normal, 10, "Normal Histrogram")


def random_normal():
    """
    Returns random draw from Standard Noral distribution
    """
    return inverse_normal_cdf(random.random())

xs = [random_normal() for _ in range(1000)]
ys1 = [x + random_normal() / 2 for x in xs]
ys2 = [-x + random_normal() / 2 for x in xs]


def scatter():
    """
    Create Scatter plot
    """
    plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
    plt.scatter(xz, ys2, marker='.', color='gray', label='ys2')
    plt.xlabel('xs')
    plt.ylabel('ys')
    plt.legend(loc=9)
    plt.show()


def correlation_matrix(data):
    """
    Returns the num_columns x num_columns matrix  whose (i. j)th entry
    is the correlation between columns i and j of data"
    """

    _, num_columns = shape(data)

    def matrix_entry(i, j):
        return correlation(get_column(data, i), get_column(data, j))

    return make_matrix(num_clumns, num_columns, matrix_entry)


def make_scatterplot_matrix():
    #  Generate Random data
    num_points = 100

    def random_row():
        row = [None, None, None, None]
        row[0] = random_normal()
        row[1] = -5 * row[0] + random_normal()
        row[2] = row[0] + row[1] + 5 + random_normal()
        row[3] = 6 if row[2] > -2 else 0
        return row
    random.seed(11)
    data = [random_row() for _ in range(num_points)]

    ## Plot it!

    _, num_columns = shape(data)
    fig, ax = plt.subplots(num_columns, num_columns)

    for i in range(num_columns):
        for j in range(num_columns):



if __name__ == "__main__":
    pass
