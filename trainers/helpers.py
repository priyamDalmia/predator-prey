import argparse

# Setting up arguments to be parsed
parser = argparse.ArgumentParser(description="Test game file", 
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--logfile', default="logs/log.txt", type=str, metavar="", help="Log file name")
parser.add_argument('--loglevel', default=20, type=int, metavar="", help="Logging level of the program (_0s)")
parser.add_argument('-t', '--train', default=False, type=bool, metavar="", help="Train agents (boolean)")
parser.add_argument('-m', '--message', default="random", type=str, metavar="", help="Message for logger")
ARGS = parser.parse_args()


