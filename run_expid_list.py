import argparse

import fuxictr_version
from fuxictr import autotuner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="../config/tuner_config.yaml",
        help="Config file for hyperparameter tuning.",
    )
    parser.add_argument(
        "--gpu",
        nargs="+",
        default=[-1],
        help="GPU indices, use -1 for CPU.",
    )
    args = vars(parser.parse_args())

    autotuner.grid_search(args["config"], args["gpu"])
