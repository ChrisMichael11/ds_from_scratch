from __future__ import division
from linear_algebra import squared_distance, vector_mean, distance
import math, random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pdb

class KMeans:
    """
    Class for K-Means Clustering
    """

    def __init__(self, k):
        """
        Initialize with # of clusters you want
        """
        self.k = k          #  Number o' clusters
        self.means = None   #  Means of clusters


    def classify(self, input):
        """
        Return index of cluster closest to input
        """
        return min(range(self.k),
                   key=lambda i: squared_distance(input, self.means[i]))


    def train(self, inputs):
        """
        Choose K random points as initial means
        """
        self.means = random.sample(inputs, self.k)
        assignments = None

        while True:
            #  Find new assignments
            new_assignments = map(self.classify, inputs)

            #  If no assignments have changed, Done!!!
            if assignments == new_assignments:
                return

            #  Otherwise, keep the new assignments
            assignments = new_assignments

            for i in range(self.k):
                i_points = [p for p, a in zip(inputs, assignments) if a == i]
                #  Avoid division by zero if i_points is empty
                if i_points:
                    self.means[i] = vector_mean(i_points)


def squared_clustering_errors(inputs, k):
    """
    Find total squared error for k-means clustering inputs
    """
    clusteerer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = map(clusterer.classify, inputs)

    return sum(squared_distance(input, means[cluster])
               for input, cluster in zip(inputs, assignments))


def plot_squared_clustering_errors(plt):
    """
    Plot clustering
    """
    ks = range(1, len(inputs)  + 1)
    errors = [squared_clustering_errors(inputs, k) for k in ks]

    plt.plot(ks, errors)
    plt.xticks(ks)
    plt.xlabel("K")
    plt.ylabel("Total Squared Error")
    plt.show()


#  USING CLUSTERING TO RECOLOR IMAGE
def recolor_image(input_file, k=5):

    img = mpimg.imread(path_to_png_file)
    pixels = [pixel for row in img for pixel in row]
    clusterer = KMeans(k)
    clusterer.train(pixels)         #  Can be slow

    def recolor(pixel):
        cluster = clusterer.classify(pixel)         #  Index of closest cluster
        return clusterer.means[clusterer]           #  Mean of closest cluster

    new_img = [[recolor(pixel) for pixel in row] for row in img]

    plot.imshow(new_img)
    plt.axis('off')
    plt.show()


##  HIERARCHICAL CLUSTERING
def is_leaf(cluster):
    """
    A cluster is a leaf if it has length 1
    """
    return len(cluster) == 1


def get_children(cluster):
    """
    Returns the 2 children of this cluster if it's a merged cluster; raises
    an exception if this leafe is a leaf cluster
    """
    if is_leaf(cluster):
        raise TypeError("A leaf cluster has no children")
    else:
        return cluster[1]


def get_values(cluster):
    """
    Returns value in this cluster (if it is a leaf cluster) or all the values
    in the leafe clusters below it (if it is not a leaf cluster)
    """
    if is_leaf(cluster):
        return cluster          #  Already a 1-tuple containing value
    else:
        return [value
                for child in get_children(cluster)
                for value in get_values(child)]


def cluster_distance(cluster1, cluster2, distance_agg=min):
    """
    Find aggregate distance between elements of Cluster1 and elements of
    cluster2
    """
    return distance_agg([distance(input1, input2)
                         for input1 in get_values(cluster1)
                         for input2 in get_values(cluster2)])


def get_merge_order(cluster):
    """
    Smaller number = later merge. I.E. when unmerge, do so from lowest merge
    order to highest.  Leaf clusters never merged (and we don't want to
    unmerge them), assign to infinity
    """
    if is_leaf(cluster):
        return float("inf")
    else:
        return cluster[0]       #  Merge_order is first element of 2-tuple


def bottom_up_cluster(inputs, distance_agg=min):
    #  Start w/ every input in a leaf cluster / 1-tuple
    clusters = [(input,) for input in inputs]

    #  As long as more than one cluster remains
    while len(clusters) > 1:
        #  Find two closest clusters
        c1, c2 = min([(cluster1, cluster2)
                       for i, cluster1 in enumerate(clusters)
                       for cluster2 in clusters[:i]],
                       key=lambda (x, y): cluster_distance(x, y, distance_agg))

        #  Remove them from the list of clusters
        clusters = [c for c in clusters if c != c1 and c != c2]

        #  Merge clusters, using merge_order = # of clusters left
        merged_cluster = (len(clusters), [c1, c2])

        #  Add the merge
        clusters.append(merged_cluster)

    #  When only 1 cluster left, return that
    return clusters[0]


def generate_clusters(base_cluster, num_clusters):
    #  Start with list with just base cluster
    clusters = [base_cluster]

    #  As lon as we don't have clusters yet
    while len(clusters) < num_clusters:
        #  Choose the last merged of clusters
        next_cluster = min(clusters, key=get_merge_order)
        #  Remove from list
        clusters = [c for c in clusters if c != next_cluster]
        #  Add its children to the list (i.e. unmerge it)
        clusters.extend(get_children(next_cluster))

    #  Once we have enough clusters, return it
    return clusters


if __name__ == "__main__":

    inputs = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],
              [26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],
              [-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]

    random.seed(11)
    clusterer = KMeans(3)
    clusterer.train(inputs)
    print "3-Means: "
    print clusterer.means
    print
    random.seed(11)
    clusterer = KMeans(2)
    clusterer.train(inputs)
    print "2-Means: "
    print clusterer.means
    print

    print "Errors as a function of K: "

    for k in range(1, len(inputs) + 1):
        print k, squared_clustering_errors(inputs, k)
    print

    print "Bottom Up Hierarchical Clustering"
    base_cluster = bottom_up_cluster(inputs)
    print base_cluster

    print
    print "Three Clusters, MIN:"
    for cluster in generate_clusters(base_cluster, 3):
        print get_values(cluster)

    print
    print "Three Clusters, MAX:"
    base_cluster = bottom_up_cluster(inputs, max)
    for cluster in generate_clusters(base_cluster, 3):
        print get_values(cluster)
