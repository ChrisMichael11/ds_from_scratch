from __future__ import division
from collections import Counter
from linear_algebra import distance, vector_subtract, scalar_multiply
import math, random


def sum_of_squares(v):
    """
    Computes sum of squared elements in v
    """
    return sum(v_i ** 2 for v_i in v)

def difference_quotient(f, x, h):
    """
    Compute difference quotient for function of single variable
    """
    return (f(x+h) - f(x)) / h


def plot_estimated_derivative():
    """
    Plot derivative
    """
    def square(x):
        """
        Square a value
        """
        return x * x

    def derivative(x):
        """
        Derivative of square function
        """
        return 2 * x

    derivative_estimate = lambda x: difference_quotient(square, x, h = 0.00001)

    #  Plot to show similar
    import matplotlib.pyplot as plt
    x = range(-10, 10)
    plt.plot(x, map(derivative, x), 'rx')           #  RED x
    plt.plot(x, map(derivative_estimate, x), 'b+')  #  BLUE +
    plt.title("Actual Derivative vs Estimated Derivative")
    plt.show()

def partial_difference_quotient(f, v, i, h):
    """
    Compute differnce quotient for function of multiple variables, f @ v
    """
    w = [v_j + (h if j == i else 0) #  Add h to only i-th element of v
        for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h

def estimate_gradient(f, v, h=0.00001):
    """
    Esitmate gradient using partial_difference_quotient
    """
    return [partial_difference_quotient(f, v, i, h) for i, _ in enumerate(v)]

##  USING GRADIENT
def step(v, direction, step_size):
    """
    Move step_size in direction from v
    """
    return [v_i + step_size * direction_i
                for v_i, direction_i in zip(v, direction)]

def sum_of_squares_gradient(v):
    """
    Compute sum of squares for v
    """
    return [2 * v_i for v_i in v]


def safe(f):
    """
    Define function that wraps f and returns it
    """
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')
    return safe_f


##  MAXIMIZE / MINIMIZE BATCH
def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """
    Use Gradient Descent to find a theta that minimizes the target function
    """
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    theta = theta_0                         #  Set theta to initial value
    target_fn = safe(target_fn)             #  Save version of target_fn
    value = target_fn(theta)                #  Value minimizing

    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size)
                            for step_size in step_sizes]

        #  Choose theta that minimizes error function
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)

        #  Stop if converging
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value


def negate(f):
    """
    Return function that returns -f(x) for input f(x)
    """
    return lambda *args, **kwargs: -f(*args, **kwargs)


def negate_all(f):
    """
    Return list that is -[...] of input [...]
    """
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]


def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """
    Maximize function, i.e. minimize its negative!!!
    """
    return minimize_batch(negate(target_fn),
                          negate_all(gradient_fn),
                          theta_0,
                          tolerance)


##  STOCHASTIC GRADIENT DESCENT
def in_random_order(data):
    """
    Generator that returns elements of array in random order
    """
    indexes = [i for i, _ in enumerate(data)]       #  Create list of indexes
    random.shuffle(indexes)                         #  Shuffle!
    for i in indexes:
        yield data[i]


def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    """
    Could possibly circle minimum forever, so when stop seeing improvement
    will decrease step_size and eventually quit
    """
    data = zip(x, y)
    theta = theta_0                                 #  Initial guess
    alpha = alpha_0                                 #  Initial step_size
    min_theta, min_value = None, float('inf')       #  Minimum so far
    iterations_with_no_improvement = 0

    #  If go 100 iterations w/no improvment, stop
    while iterations_with_no_improvement < 100:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)

        if value < min_value:
            #  If find new minimum, remember it
            #  Go back to OG step_size
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            #  Otherwies, not improving, shrink step_size
            iterations_with_no_improvement += 1
            alpha *= 0.9

        #  Take gradient step for each data points
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))

    return min_theta


def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0 = 0.01):
    """
    Same as minimize, but negate
    """
    return minimize_stochastic(negate(target_fn),
                               negate_all(gradient_fn),
                               x, y, theta_0, alpha_0)




if __name__ == "__main__":
    plot_estimated_derivative()
    print "using the gradient"

    v = [random.randint(-10,10) for i in range(3)]

    tolerance = 0.0000001

    while True:
        #print v, sum_of_squares(v)
        gradient = sum_of_squares_gradient(v)   #  Compute the gradient at v
        next_v = step(v, gradient, -0.01)       #  Take negative gradient step
        if distance(next_v, v) < tolerance:     #  Stop if converging
            break
        v = next_v                              #  Continue if not converging

    print "minimum v", v
    print "minimum value", sum_of_squares(v)
    print


    print "using minimize_batch"

    v = [random.randint(-10,10) for i in range(3)]

    v = minimize_batch(sum_of_squares, sum_of_squares_gradient, v)

    print "minimum v", v
    print "minimum value", sum_of_squares(v)
