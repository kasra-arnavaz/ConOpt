import sys

sys.path.append("src")

from mesh.mesh_factory import MeshFactoryFromScad
from mesh.scad import Scad
import os
import argparse


def main(args):
    scad = Scad(file=args.scad_file, parameters=args.scad_params)
    MeshFactoryFromScad(scad=scad, ideal_edge_length=args.edge_length).create()
    os.rename(".tmp/mesh.msh", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scad_file", type=str, default="data/caterpillar.scad", help="path like to .scad")
    parser.add_argument(
        "--scad_params", type=str, default="data/caterpillar_scad_params.json", help="path like to .json"
    )
    parser.add_argument("--edge_length", type=float, default=0.02)
    parser.add_argument("--out", type=str, help="path like to where to save .msh")
    args = parser.parse_args()
    main(args)
