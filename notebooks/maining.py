import argparse
from ast import parse
import os
import numpy as np

def parse_args(args):
    parser = argparse.ArgumentParser(description='Test Argument Parse')
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--opt1", default=False)
    parser.add_argument("--opt2", default='Optional Test')
    parser.add_argument("--opt3", type=int, default=3)
    return parser.parse_args(args)




def main(args):
    args = parse_args(args)
    print(args)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])