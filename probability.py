from __future__ import division
from collections import Counter
import math, random
import matplotlib.pyplot as plt


def random_kid():
    """
    Generate "boy" or "girl" randomly
    """
    return random.choice(["boy", "girl"])


def uniform_pdf(x):
    """
    Generate density function for uniform distribution
    """
    return 1 if x >= 0 and x < 1 else 0


def uniform_cdf(x):
    """
    return probaility that uniform random variable is <= x
    """
    if x < 0: return 0          #  Uniform random never less than 0
    elif x < 1: return x        #  ex P(X <= 0.3) = 0.3
    else: return 1              #  Uniform random always less than 1


def normal_pdf(x, mu=0, sigma=1):
    """
    Generate normal distribution
    """
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))

def plot_normal_pdfs(plt):
    """
    Plot normal pdf
    """
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs], '-', label='mu = 0, sigma = 1')
    plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs], '-', label='mu = 0, sigma = 2')
    plt.plot(xs, [normal_pdf(x, sigma=0.5) for x in xs], '-', label='mu = 0, sigma = 0.5')
    plt.plot(xs, [normal_pdf(x, mu=1) for x in xs], '-', label='mu = 1, sigma = 1')
    plt.legend()
    plt.title('Normal PDFs')
    plt.show()


def normal_cdf(x, mu=0, sigma=1):
    """
    Define normal cdf
    """
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2


def plot_normal_cdfs(plt):
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu = 0,sigma = 1')
    plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu = 0,sigma = 2')
    plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mu = 0,sigma = 0.5')
    plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu = -1,sigma = 1')
    plt.legend(loc=4)               # Bottom Right
    plt.title('Normal CDFs')
    plt.show()


def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    """
    Find value corresponding to specified probability
    Find approx inverse using binary search
    """
    #  If inputs are not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z, low_p = -10.0, 0         #  normal_cdf(-10) is really close to 0
    hi_z, hi_p = 10.0, 1            #  normal_cdf(10) is really close to 1

    while hi_z - low_z > tolerance:
        mid_z = (hi_z + low_z) / 2  #  midpoint
        mid_p = normal_cdf(mid_z)   #  CDF at midpoint

        if mid_p < p:
            #  Midpoint too low, search above it
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            #  midpoint too high, search below
            hi_z, hi_p = mid_z, mid_p
        else:
            break
    return mid_z


def bernoulli_trial(p):
    """
    Perform bernoulli trial for p = 1, 0 = p-1
    """
    return 1 if random.random() < p else 0

def binomial(n, p):
    """
    Perfrom binomial (sum of n random Bernolli(p) trials)
    """
    return sum(bernoulli_trial(p) for _ in range(n))

def make_hist(p, n, num_points):
    """
    Make histogram
    """
    data = [binomial(n, p) for _ in range(num_points)]

    #  Use bar chart to show binomial samples
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8, color='0.75')

    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))

    #  Use line chart to show normal approximation
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma) for i in xs]
    plt.plot(xs, ys)
    plt.title('Binomial Distribution vs Normal Approximation')
    plt.show()


if __name__ == "__main__":
    # CONDITIONAL PROBABILITY

    both_girls = 0
    older_girl = 0
    either_girl = 0

    random.seed(11)
    for _ in range(10000):
        younger = random_kid()
        older = random_kid()
        if older == "girl":
            older_girl += 1
        if older == "girl" and younger == "girl":
            both_girls += 1
        if older == "girl" or younger == "girl":
            either_girl += 1

    print "P(both | older):", both_girls / older_girl      # 0.514 ~ 1/2
    print "P(both | either): ", both_girls / either_girl   # 0.342 ~ 1/3
    plot_normal_pdfs(plt)
    plot_normal_cdfs(plt)
    make_hist(0.69, 100, 10000)
