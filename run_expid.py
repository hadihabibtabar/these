import argparse
import gc
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import fuxictr_version
from fuxictr import datasets
from fuxictr.features import FeatureMap
from fuxictr.preprocess import FeatureProcessor, build_dataset
from fuxictr.pytorch.dataloaders import H5DataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.utils import load_config, print_to_json, print_to_list, set_logger

import src as model_zoo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/", help="Config directory")
    parser.add_argument("--expid", type=str, required=True, help="Experiment id")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index, -1 for CPU")
    args = vars(parser.parse_args())

    experiment_id = args["expid"]
    params = load_config(args["config"], experiment_id)
    params["gpu"] = args["gpu"]
    set_logger(params)
    logging.info("Params: %s", print_to_json(params))
    seed_everything(seed=params["seed"])

    data_dir = os.path.join(params["data_root"], params["dataset_id"])
    feature_map_json = os.path.join(data_dir, "feature_map.json")

    if params["data_format"] == "csv":
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = build_dataset(
            feature_encoder, **params
        )

    feature_map = FeatureMap(params["dataset_id"], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: %s", print_to_json(feature_map.features))

    model_class = getattr(model_zoo, params["model"])
    model = model_class(feature_map, **params)
    model.count_parameters()

    train_gen, valid_gen = H5DataLoader(feature_map, stage="train", **params).make_iterator()
    model.fit(train_gen, validation_data=valid_gen, **params)

    logging.info("Validation evaluation")
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()

    result_filename = Path(args["config"]).name.replace(".yaml", "") + ".csv"
    with open(result_filename, "a+", encoding="utf-8") as result_file:
        result_file.write(
            " {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {} \n".format(
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                " ".join(sys.argv),
                experiment_id,
                params["dataset_id"],
                "N.A.",
                print_to_list(valid_result),
            )
        )
