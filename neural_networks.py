from __future__ import division
from collections import Counter
from functools import partial
from linear_algebra import dot
import math, random
import matplotlib.pyplot as plt

import matplotlib

def step_function(x):
    """
    Define step function for NN nodes
    """
    return 1 if x >= 0 else 0


def perceptron_output(weights, bias, x):
    """
    Returns 1 if perceptron "fires", 0 if it doesn't
    """
    calculation = dot(weights, x) + bias
    return step_function(calculation)


def sigmoid(t):
    """
    Define sigmoid function for NN nodes for smooth approximation of step function
    """
    return 1 / (1 + math.exp(-t))


def neuron_output(weights, inputs):
    """
    Define output of neuron using sigmoid function
    """
    return sigmoid(dot(weights, inputs))


def feed_forward(neural_network, input_vector):
    """
    Take a NN (list (layers) of lists (neurons) of lists (weights)) and returns
    output from foward propagating the input
    """
    outputs = []

    for layer in neural_network:

        input_with_bias = input_vector + [1]                #  Add bias input
        output = [neuron_output(neuron, input_with_bias)    #  Compute output
                                for neuron in layer]        #  For each later
        outputs.append(output)                              #  Remember output

        input_vector = output

    return outputs


def backpropagate(network, input_vector, target):

    hidden_outputs, outputs = feed_forward(network, input_vector)

    #  Output * (1 0 output) is derivative of sigmoid!!!
    output_deltas = [output * (1 - output) * (output - target[i])
                        for i, output in enumerate(outputs)]

    #  Adjust weights for output later (network[-1])
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output

    #  Backpropagate errors to hidden layer
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                     dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    #  Adjust weights f or hidden layer (network[0])
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] + input

def patch(x, y, hatch, color):
    """
    Return matplotlib "patch" object with location, pattern and color
    """
    return matplotlib.patches.Rectangle((x - 0.5, y - 0.5), 1, 1, hatch=hatch,
                                        fill=False, color=color)


def show_weights(neuron_idx):
    """
    Show weights for neuron_idx
    """
    weights = netowrk[0][neuron_idx]
    abs_weights = maps(abs, weights)

    grid = [abs_weights[row: (row + 5)]         #  Turn on weights in 5 x 5 grid
                for row in range(0, 25, 5)]     #  Weights[0:5], ..., weights[20:25]

    ax = plt.gca()                              #  To use hatching need axis

    ax.imshow(grid,                             #  Same as plt.imshow
              cmap=matplotlib.cm.binary,        #  Use white/black colorscale
              interpolation='none')             #  Plot blocks as blocks

    #  Crosshatch negative weights
    for i in range(5):                          #  Row
        for j in range(5):                      #  Column
            if weights[5 * i + j] < 0:          #  Row i, Column j = weights [5 * i + j]
                #  Add B/W hatches
                ax.add_patch(patch(j, i, '/', 'white'))
                ax.add_patch(patch(j, i, '\\', 'black'))
    plt.show()



if __name__ == "__main__":

    raw_digits = [
            # 0
          """11111
             1...1
             1...1
             1...1
             11111""",
             # 1
          """..1..
             ..1..
             ..1..
             ..1..
             ..1..""",
             # 2
          """11111
             ....1
             11111
             1....
             11111""",
             # 3
          """11111
             ....1
             11111
             ....1
             11111""",
             # 4
          """1...1
             1...1
             11111
             ....1
             ....1""",
             # 5
          """11111
             1....
             11111
             ....1
             11111""",
             # 6
          """11111
             1....
             11111
             1...1
             11111""",
             # 7
          """11111
             ....1
             ....1
             ....1
             ....1""",
             # 8
          """11111
             1...1
             11111
             1...1
             11111""",
             # 9
          """11111
             1...1
             11111
             ....1
             11111"""]

    def make_digit(raw_digit):
        return [1 if c == '1' else 0
                for row in raw_digit.split("\n")
                for c in row.strip()]

    inputs = map(make_digit, raw_digits)

    targets = [[1 if i == j else 0 for i in range(10)]
               for j in range(10)]

    random.seed(11)
    input_size = 25  #  Each input is a vector of length 25
    num_hidden = 5   #  5 neurons in the hidden layer
    output_size = 10 #  Need 10 outputs for each input

    #  Each hidden neuron has one weight per input, plus bias weight
    hidden_layer = [[random.random() for __ in range(input_size + 1)]
                    for __ in range(num_hidden)]

    #  Each output neuron has one weight per hidden neuron, plus bias weight
    output_layer = [[random.random() for __ in range(num_hidden + 1)]
                    for __ in range(output_size)]

    #  The network starts out with random weights
    network = [hidden_layer, output_layer]

    #  Try 10,000 iterations
    for __ in range(10000):
        for input_vector, target_vector in zip(inputs, targets):
            backpropagate(network, input_vector, target_vector)

    def predict(input):
        return feed_forward(network, input)[-1]

    for i, input in enumerate(inputs):
        outputs = predict(input)
        print i, [round(p,2) for p in outputs]

    print """.@@@.
...@@
..@@.
...@@
.@@@."""
    print [round(x, 2) for x in
          predict(  [0,1,1,1,0,  # .@@@.
                     0,0,0,1,1,  # ...@@
                     0,0,1,1,0,  # ..@@.
                     0,0,0,1,1,  # ...@@
                     0,1,1,1,0]) # .@@@.
          ]
    print

    print """.@@@.
@..@@
.@@@.
@..@@
.@@@."""
    print [round(x, 2) for x in
          predict(  [0,1,1,1,0,  # .@@@.
                     1,0,0,1,1,  # @..@@
                     0,1,1,1,0,  # .@@@.
                     1,0,0,1,1,  # @..@@
                     0,1,1,1,0]) # .@@@.
          ]
    print
