from __future__ import division
import math, random, re
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt

def plot_resumes(plt):
    """
    Plot resumes word cloud
    """
    data = [ ("big data", 100, 15), ("Hadoop", 95, 25), ("Python", 75, 50),
         ("R", 50, 40), ("machine learning", 80, 20), ("statistics", 20, 60),
         ("data science", 60, 70), ("analytics", 90, 3),
         ("team player", 85, 85), ("dynamic", 2, 90), ("synergies", 70, 0),
         ("actionable insights", 40, 30), ("think out of the box", 45, 10),
         ("self-starter", 30, 50), ("customer focus", 65, 15),
         ("thought leadership", 35, 35)]

    def text_size(total):
        """
        Size text modifier.  8 if total - 0, 28 if total = 200
        """
        return 8 + total / 200 * 20

    for word, job_popularity, resume_popularity in data:
        plt.text(job_popularity, resume_popularity, word, ha='center',
                 va='center',
                 size=text_size(job_popularity + resume_popularity))
    plt.xlabel("Popularity on Job Postings")
    plt.ylabel("Popularity on Resumes")
    plt.axis([0, 100, 0, 100])
    plt.show()


## N-GRAM models
def fix_unicode(text):
    """
    Fix unicode apostrophes with normal ones
    """
    return text.replace(u"\u2019", "'")


def get_document():
    """
    Get data (text) from webpage and split into a sequence of words and periods
    (to find sentence ends)  re.findall() does the dirty work
    """
    url = "http://radar.oreilly.com/2010/06/what-is-data-science.html"
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html5lib")          #  Find Article/Body div
    regex = r"[\w']+|[\.]"                          #  Matches word or period

    document = []

    for paragraph in content("p"):
        words = re.finall(regex, fix_unicode(paragraph.text))
        document.extend(words)

    return document


def generage_using_bigrams(transitions):
    """
    Create pairs of words.  Zip stores consecutive elements!

    bigrams = zip(document, document[1:])
    transitions = defaultdict(list)
    for prev, current in bigrams:
        transitions[prev].append(current)
    """
    current = "."           #  Next world will start a sentence
    result = []
    while True:
        next_word_candidates = transitions[current]     #  bigrams (current, _)
        current = random.choice(next_word_candidates)   #  Choose on at random
        result.append(current)                          #  Append to results
        if current == ".": return " ".join(result)      #  If ".", DONE!

def generate_using_trigrams(starts, trigram_transitions):
    """
    Create triplets of words.

    trigrams = zip(document, document[1:], document[2:])
    trigram_transitions = defaultdict(list)
    starts = []

    for prev, current, next in trigrams:

        if prev == ".": # if the previous "word" was a period
            starts.append(current) # then this is a start word

        trigram_transitions[(prev, current)].append(next)
    """
    current = random.choice(starts)         #  Random start word
    prev = "."                              #  Preced with '.'
    result = [current]
    while True:
        next_word_candidates = trigram_transitions[(prev, current)]
        next = random.choice(next_word_candidates)

        prev, current = current, next
        result.append(current)

        if current == ".":
            return " ".join(results)

def is_terminal(token):
    """
    Helper function to determine if token is terminal
    """
    return token[0] != "_"


def expand(grammar, tokens):
    for i, token in enumerate(tokens):

        #  Ignore terminals
        if is_terminal(token): continue

        #  CHoose replacement at random
        replacement = random.choice(grammar[token])

        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            tokens = tokens[:i] + replacement.split() + tokens[(i + 1):]
        return expand(grammar, tokens)

    #  If we get here, we had all terminals and we are done
    return tokens

def generate_sentence(gramamr):
    """
    Generate sentences from data(text)
    """
    return expand(grammar, ["_S"])


##  GIBBS SAMPLING
def roll_a_die():
    """
    Simulate 6 sided fair die
    """
    return random.choice(1, 2, 3, 4, 5, 6)


def direct_sample():
    """
    Sample roll of 2 die
    """
    d1 = roll_a_die()
    d2 = roll_a_die()
    return d1, d1 + d2


def random_y_given_x(x):
    """
    Return y conditional on x
    Equally likely to be x + 1, x + 2,...,x + 6
    """
    return x + roll_a_die()


def random_x_given_y(y):
    """
    Return x given y.
    Examples:
    y = 2, then x = 1
    y = 3, then x = 1 or 2 (equally likely)
    y = 11, then x = 5 or 6 (equally likely)
    """
    if y <= 7:
        #  If total is 7 or less, first die is equally likely to be
        #  1, 2,..., (total - 1)
        return random.randrange(1, y)

    else:
        #  If total is 7 or more, the first die is equally likely to be
        #  (total - 6), (total - 5), ..., 6
        return random.randrange(y -6, 7)

def gibbs_sample(num_iters=1000):
    """
    Start with any (valid) value for x and y and then repeatedly alternate
    replacing x with a random value picked conditional on y and replacing y
    with a random value picked conditional on x. After a number of iter‐ ations,
    the resulting values of x and y will represent a sample from the
    unconditional joint distribution
    """
    x, y = 1, 2         #  Could be whatever, not too important
    for _ in range(num_iters):
        x = random_x_given_y(y)
        y = random_y_given_x(x)
    return x, y


def compare_distributions(num_samples=1000):
    """
    Check that gibbs sampling gives similar results to direct sample
    """
    counts = defaultdict(lambda: [0, 0])
    for _ in range(num_samples):
        counts[gibbs_sample()][0] += 1
        counts[gibbs_sample()][1] += 1
    return counts


#  TOPIC MODELING




if __name__ == "__main__":
    plot_resumes(plt)           #  Super-Awesome Word Cloud!
