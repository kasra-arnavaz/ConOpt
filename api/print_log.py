import sys
import argparse

sys.path.append("src")

from objective.log import Log


def main(args):
    if args.iter is None:
        Log.print(path=args.path)
    else:
        Log.print_iteration(path=args.path, i=args.iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to logs")
    parser.add_argument("-i", "--iter", type=int, nargs="?", default=None, help="iteration to print")
    args = parser.parse_args()
    main(args)
