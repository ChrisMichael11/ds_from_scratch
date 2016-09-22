from __future__ import division
from probability import normal_cdf, inverse_normal_cdf
import math, random

def normal_approximation_to_binomial(n, p):
    """
    Finds mu and sigma corresponding to binomial(n, p)
    """
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

##  PROBABILITY A RANDOM VARIABLE FOLLOWS NORMAL DISTRIBUTION AND IS WITHIN (OR OUTSIDE)
##  A PARTICULAR INTERVAL

#  Normal cdf is the probability the varaible is below a threshold
normal_probability_below = normal_cdf

def normal_probability_above(lo, mu=0, sigma=1):
    """
    Probability is above the threshold if it is not below
    """
    return 1 - normal_cdf(lo, mu, sigma)

def normal_probability_between(lo, hi, mu=0, sigma=1):
    """
    Probability is between if it is less than hi, but not less than lo
    """
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

def normal_probability_outside(lo, hi, mu=0, sigma=1):
    """
    It is outside if not between
    """
    return 1 - normal_probability_between(lo, hi, mu, sigma)

##  REVERSE OF ABOVE, FIND INTERVAL AROUND
##  FIND INTERVAL CENTERED AT MEAN, CONTAINING 60% PROBABILITY, FIND CUTOFFS WHERE
##  UPPER AND LOWER TAILS EACH CONTAIN 20% OF PROBABILITY

def normal_upper_bound(probability, mu=0, sigma=1):
    """
    Returns z for which P(Z <= z) = probability
    """
    return inverse_normal_cdf(probability, mu, sigma)


def normal_lower_bound(probability, mu=0, sigma=1):
    """
    Returns z for which P(Z >= z) = probabitily
    """
    return inverse_normal_cdf(1 - probability, mu, sigma)


def normal_two_sided_bounds(probability, mu=0, sigma=1):
    """
    Returns symmetric (about the mean) bounds that contain specified probability
    """
    tail_probability = (1 - probability) / 2

    #  Upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    #  Lower bouund should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

def two_sided_p_value(x, mu=0, sigma=1):
    if x >= mu:
        #  if x is greater than mean, tail is above x
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        return 2 * normal_probability_below(x, mu, sigma)


def count_extreme_values():
    extreme_value_count = 0
    for _ in range(100000):
        num_heads = sum(1 if random.random() < 0.5 else 0    #  Count # of heads
                        for _ in range(1000))                #  In 1000 flips
        if num_heads >= 530 or num_heads <= 470:             #  And count how often
            extreme_value_count += 1                         #  The # is 'extreme'

    return extreme_value_count / 100000

upper_p_value = normal_probability_above
lower_p_value = normal_probability_below

## P-hacking
def run_experiment():
    """
    Flip fair coin 1000 times.  True = H, False = T
    """
    return [random.random() < 0.5 for _ in range(1000)]


def reject_fairness(experiment):
    """
    Use 5% significance levels
    """
    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531

## A/B TESTING
def estimated_parameters(N, n):
    """
    N = views, n = clicks
    """
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma


def a_b_test_statistic(N_A, n_A, N_B, n_B):
    """
    Test null hypothesis
    """
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B **2)


##  BAYESIAN INFERENCE
def B(alpha, beta):
    """
    Normalizing constant, make probability = 1
    """
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)


def beta_pdf(x, alpha, beta):
    """
    Create beta PDF
    NOTE - generally centered at alpha / (alpha + beta)
    """
    if x < 0 or x > 1:
        return 0            #  Gotta be between [0, 1]
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)


if __name__ == "__main__":

    mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
    print "mu_0", mu_0
    print "sigma_0", sigma_0
    print "normal_two_sided_bounds(0.95, mu_0, sigma_0)", normal_two_sided_bounds(0.95, mu_0, sigma_0)
    print
    print "power of a test"

    print "95% bounds based on assumption p is 0.5"

    lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)
    print "lo", lo
    print "hi", hi

    print "actual mu and sigma based on p = 0.55"
    mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)
    print "mu_1", mu_1
    print "sigma_1", sigma_1

    # a type 2 error means we fail to reject the null hypothesis
    # which will happen when X is still in our original interval
    type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
    power = 1 - type_2_probability # 0.887

    print "type 2 probability", type_2_probability
    print "power", power
    print

    print "one-sided test"
    hi = normal_upper_bound(0.95, mu_0, sigma_0)
    print "hi", hi # is 526 (< 531, since we need more probability in the upper tail)
    type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
    power = 1 - type_2_probability # = 0.936
    print "type 2 probability", type_2_probability
    print "power", power
    print

    print "two_sided_p_value(529.5, mu_0, sigma_0)", two_sided_p_value(529.5, mu_0, sigma_0)

    print "two_sided_p_value(531.5, mu_0, sigma_0)", two_sided_p_value(531.5, mu_0, sigma_0)

    print "upper_p_value(525, mu_0, sigma_0)", upper_p_value(525, mu_0, sigma_0)
    print "upper_p_value(527, mu_0, sigma_0)", upper_p_value(527, mu_0, sigma_0)
    print

    print "P-hacking"

    random.seed(11)
    experiments = [run_experiment() for _ in range(1000)]
    num_rejections = len([experiment
                          for experiment in experiments
                          if reject_fairness(experiment)])

    print num_rejections, "rejections out of 1000"
    print

    print "A/B testing"
    z = a_b_test_statistic(1000, 200, 1000, 180)
    print "a_b_test_statistic(1000, 200, 1000, 180)", z
    print "p-value", two_sided_p_value(z)
    z = a_b_test_statistic(1000, 200, 1000, 150)
    print "a_b_test_statistic(1000, 200, 1000, 150)", z
    print "p-value", two_sided_p_value(z)
