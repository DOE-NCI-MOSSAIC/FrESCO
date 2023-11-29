#!/usr/bin/env python3

"""
    Top level module to create information extraction or case-level context models.

"""

import argparse
import os
import time

import run_ie
import run_clc

from validate import exceptions


def timestamp():
    t = time.time()
    print(time.ctime(t), "\n", flush=True)

    return t


def main():
    print("\nProcess started")
    start = timestamp()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="ie",
        help="""which type of model to create. Must be either
                                IE (information extraction) or clc (case-level context).""",
    )
    parser.add_argument(
        "--model_path",
        "-mp",
        type=str,
        default="",
        help="""this is the location of the model
                               that will used to make predictions""",
    )
    parser.add_argument(
        "--data_path",
        "-dp",
        type=str,
        default="",
        help="""where the data will load from. The default is
                                the path saved in the model""",
    )
    parser.add_argument(
        "--model_args",
        "-args",
        type=str,
        default="",
        help="""file specifying the model or clc args; default is in
                                the model_suite directory""",
    )

    args = parser.parse_args()

    if len(args.model) == 0:
        raise exceptions.ParamError("Must specify either ie or clc model to build. Exiting")

    if not os.path.exists("predictions"):
        os.makedirs("predictions")

    if args.model == "ie":
        print("Running information extraction task")
        run_ie.run_ie(args)
    elif args.model == "clc":
        print("Running case level context model")
        run_clc.run_case_level(args)
    else:
        raise exceptions.ParamError("Must specify either 'ie' or 'clc' model to build. Exiting")

    end = timestamp()
    print("\nProcess concluded")
    print(f"Total time elapsed: {end - start:.2f}")


if __name__ == "__main__":
    main()
