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
            #  Scatter column_j on the x-axis vs column_i on the y-axis
            if i != j: ax[i][j].scatter(get_coulmn(data, j), get_column(data, i))

            #  Unless i == j, then show series
        else: ax[i][j].annotate("Series " + str(i), (0.5, 0.5),
                                xycoords='Axes Fraction',
                                ha="center", va="center")

        #  Hide axis labels, except at left and bottom charts
        if i < num_columns - 1: ax[i][j].xaxis.set_visible(False)
        if j > 0: ax[i][j].yaxis.set_visible(False)

    #  Fix bottom right and top left axis lables, which are wrong
    #  Chars only ahve text!
    ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
    ax[0][0].set_ylim(ax[0][1].gt_ylim())

    plt.show()

def parse_row(input_row, parsers):
    """
    Given list of parsers (can be None), apply appropriate one to each element
    of input_row
    """
    return [parser(value) if parser is not None else value for value, parser
                    in zip(input_row, parser)]

def parse_rows_with(reader, parsers):
    """
    Wrap a reader to apply parser to each of its rows
    """
    for row in reader:
        yield parse_row(row, parsers)

def try_or_none(f):
    """
    Wraps f to return None if f raises an exception.  Assumes f takes one input_dict
    """
    def f_or_none(x):
        try: return f(x)
        except: return None
    return f_or_none


def try_parse_field(field_name, value, parser_dict):
    """
    Try to parse value using function from parser_dict
    """
    parser = parser_dict.get(field_name)            #  Returns None, if no entry
    if parser is not None:
        return try_or_none(parser)(value)
    else:
        return value


##  MANIPULATING DATA

def picker(field_name):
    """
    returns function that picks a field out of a dict
    """
    return lambda row: row[field_name]


def pluck(field_name, rows):
    """
    Turn list of dicts into list of field_name values
    """
    return map(picker(field_name), rows)


def group_by(grouper, rows, value_transform=None):
    """
    Group rows by result, can apply values xform too
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[grouper(row)].append(row)
    if value_transform is None:
        return grouped
    else:
        return {key: value_transform(rows) for key, rows in grouped.iteritems()}


def percent_price_change(yesterday, today):
    """
    Calculate % price change from yesterday based on closing data
    """
    return today["closing_price"] / yesterday["closing_price"] - 1


def day_over_day_changes(grouped_rows):
    """
    Calculate percent price change for multiple groups
    """
    #  Sort rows by date
    ordered = sorted(grouped_rows, key=picker("date"))

    #  ZIP with an offset toget pairs of consecutive days
    return [{"symbol" : today["symbol"],
             "date" : today["date"],
             "change" : percent_price_change(yesterday, today)}
             for yesterday, today in zip(ordered, ordered[1:])]


##  RESCALING DATA


if __name__ == "__main__":
    pass
