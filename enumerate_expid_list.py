import argparse

import pandas as pd

import fuxictr_version
from fuxictr import autotuner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="../config/tuner_config.yaml",
        help="Config file for hyperparameter tuning.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="CSV file of completed experiments to exclude.",
    )
    args = vars(parser.parse_args())

    excluded = []
    if args["exclude"]:
        result_df = pd.read_csv(args["exclude"], header=None)
        excluded = result_df.iloc[:, 2].map(lambda x: x.replace("[exp_id] ", "")).tolist()

    autotuner.enumerate_params(args["config"], exclude_expid=excluded)
