import sys
import argparse

sys.path.append("src")

from objective.log import Log


def main(args):
    Log.print(path=args.path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to logs")
    args = parser.parse_args()
    main(args)
