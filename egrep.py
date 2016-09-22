import sys, re

if __name__ == "__main__":
    """
    sys.argv is the list of command line arguments
    sys.argv[0] is the name of the program itself
    sys.argv[1] will be the regex specified at the commande line
    """
    regex = sys.argv[1]

    """
    For every line passed into the script, if it matches the regex, write it out
    to stdout
    """
    for line in sys.stdin:
        if re.search(regex, line):
            sys.stdout.write(line)
