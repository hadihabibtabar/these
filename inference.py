import argparse
import logging
import os
from pathlib import Path

import pandas as pd

import fuxictr_version
from fuxictr import datasets
from fuxictr.features import FeatureMap
from fuxictr.preprocess import FeatureProcessor, build_dataset
from fuxictr.pytorch.dataloaders import H5DataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.utils import load_config, print_to_json
import src as model_zoo


def configure_inference_logger(params):
    dataset_id = params["dataset_id"]
    model_id = params.get("model_id", "")
    log_dir = os.path.join(params.get("model_root", "./checkpoints"), dataset_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, model_id + "#inference.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )


def read_submission_source(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.shape[1] == 1:
        frame = pd.read_csv(path, sep="\t")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/", help="Config directory")
    parser.add_argument("--expid", type=str, required=True, help="Experiment id")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index, -1 for CPU")
    parser.add_argument(
        "--submission_source",
        type=str,
        default=None,
        help="CSV source used to populate RowId in submission output",
    )
    parser.add_argument("--row_id_col", type=str, default="f_0", help="Row id column name")
    args = vars(parser.parse_args())

    params = load_config(args["config"], args["expid"])
    params["gpu"] = args["gpu"]
    configure_inference_logger(params)
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
    model.load_weights(model.checkpoint)
    logging.info("Loaded model checkpoint: %s", model.checkpoint)

    submission_dir = Path("./submission")
    submission_dir.mkdir(parents=True, exist_ok=True)

    submission_source = args["submission_source"] or params.get("submission_source") or params.get("test_data")
    row_ids = None
    if submission_source and os.path.exists(submission_source):
        source_frame = read_submission_source(submission_source)
        row_id_col = args["row_id_col"]
        if row_id_col in source_frame.columns:
            row_ids = source_frame[row_id_col].values
        elif len(source_frame.columns) > 0:
            row_ids = source_frame.iloc[:, 0].values

    params["shuffle"] = False
    test_gen = H5DataLoader(feature_map, stage="test", **params).make_iterator()
    y_pred_all = model.predict(test_gen)

    submission_data = pd.DataFrame()
    if row_ids is None:
        first_prediction_key = next(iter(y_pred_all.keys()))
        row_ids = range(len(y_pred_all[first_prediction_key]))
    submission_data["RowId"] = row_ids

    for key, values in y_pred_all.items():
        submission_data[key] = values

    output_path = submission_dir / f"submission_{args['expid']}.csv"
    submission_data.to_csv(output_path, index=False, sep="\t")
    logging.info("Saved submission file: %s", output_path)
