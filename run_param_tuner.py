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
        "--tag",
        type=str,
        default=None,
        help="Run only experiment IDs matching a tag.",
    )
    parser.add_argument(
        "--gpu",
        nargs="+",
        default=[-1],
        help="GPU indices, use -1 for CPU.",
    )
    args = vars(parser.parse_args())

    config_dir = autotuner.enumerate_params(args["config"])
    autotuner.grid_search(config_dir, args["gpu"], args["tag"])
