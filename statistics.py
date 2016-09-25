from __future__ import division
from collections import Counter
from linear_algebra import sum_of_squares, dot
import math
import matplotlib.pyplot as plt

num_friends = [100,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,\
                13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,\
                9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,\
                7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,\
                6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,\
                4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,\
                2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

def make_friend_counts_histogram(plt):
    friend_counts = Counter(num_friends)
    xs = range(101)
    ys = [friend_counts[x] for x in xs]
    plt.bar(xs, ys)
    plt.axis([0,101,0,25])
    plt.title("Histogram of Friend Counts")
    plt.xlabel("# of friends")
    plt.ylabel("# of people")
    plt.show()

num_points = len(num_friends)               # 204

largest_value = max(num_friends)            # 100
smallest_value = min(num_friends)           # 1

sorted_values = sorted(num_friends)
smallest_value = sorted_values[0]           # 1
second_smallest_value = sorted_values[1]    # 1
second_largest_value = sorted_values[-2]    # 49

def mean(x):



if __name__ == "__main__":

    print "num_points", len(num_friends)
    # make_friend_counts_histogram(plt)
    print "largest value", max(num_friends)             # 100
    print "smallest value", min(num_friends)            # 1
    print "second_smallest_value", sorted_values[1]     # 1
    print "second_largest_value", sorted_values[-2]     # 49
    # print "mean(num_friends)", mean(num_friends)
    # print "median(num_friends)", median(num_friends)
    # print "quantile(num_friends, 0.10)", quantile(num_friends, 0.10)
    # print "quantile(num_friends, 0.25)", quantile(num_friends, 0.25)
    # print "quantile(num_friends, 0.75)", quantile(num_friends, 0.75)
    # print "quantile(num_friends, 0.90)", quantile(num_friends, 0.90)
    # print "mode(num_friends)", mode(num_friends)
    # print "data_range(num_friends)", data_range(num_friends)
    # print "variance(num_friends)", variance(num_friends)
    # print "standard_deviation(num_friends)", standard_deviation(num_friends)
    # print "interquartile_range(num_friends)", interquartile_range(num_friends)
    #
    # print "covariance(num_friends, daily_minutes)", covariance(num_friends, daily_minutes)
    # print "correlation(num_friends, daily_minutes)", correlation(num_friends, daily_minutes)
    # print "correlation(num_friends_good, daily_minutes_good)", correlation(num_friends_good, daily_minutes_good)
    #
