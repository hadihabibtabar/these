import argparse
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from fuxictr.features import FeatureMap
from fuxictr.preprocess import FeatureProcessor, build_dataset
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.utils import load_config, load_h5, print_to_json, set_logger

import src as model_zoo
from src.fl_client import start_client
from src.fl_server import create_server, start_server


class RowTensorDataset(Dataset):
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __len__(self):
        return self.tensor.size(0)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.tensor[index]


def partition_frame_iid(df: pd.DataFrame, num_clients: int, seed: int) -> List[pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return [split.reset_index(drop=True) for split in np.array_split(shuffled, num_clients)]


def partition_frame_noniid(
    df: pd.DataFrame,
    num_clients: int,
    partition_feature: str,
    seed: int,
) -> List[pd.DataFrame]:
    if partition_feature not in df.columns:
        return partition_frame_iid(df, num_clients, seed)

    hashed = pd.util.hash_pandas_object(df[partition_feature], index=False).astype(np.int64)
    assignment = np.mod(np.abs(hashed.values), num_clients)

    partitions = [df.iloc[assignment == client_id].copy() for client_id in range(num_clients)]

    empty_clients = [idx for idx, part in enumerate(partitions) if len(part) == 0]
    if empty_clients:
        iid_backup = partition_frame_iid(df, num_clients, seed)
        for idx in empty_clients:
            partitions[idx] = iid_backup[idx]

    return [split.reset_index(drop=True) for split in partitions]


def partition_data(
    df: pd.DataFrame,
    num_clients: int,
    iid: bool,
    partition_feature: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    train_val_df, _ = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size / (1.0 - test_size),
        random_state=seed,
        shuffle=True,
    )

    splitter = partition_frame_iid if iid else partition_frame_noniid
    if iid:
        train_parts = splitter(train_df, num_clients, seed)
        val_parts = splitter(val_df, num_clients, seed)
    else:
        train_parts = splitter(train_df, num_clients, partition_feature, seed)
        val_parts = splitter(val_df, num_clients, partition_feature, seed)

    return list(zip(train_parts, val_parts))


def frame_to_dataloader(
    frame: pd.DataFrame,
    ordered_columns: List[str],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    array = frame[ordered_columns].to_numpy(dtype=np.float32)
    tensor = torch.from_numpy(array)
    dataset = RowTensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def ensure_feature_map_and_h5(params: dict) -> FeatureMap:
    if params["data_format"] == "csv":
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = build_dataset(
            feature_encoder, **params
        )

    data_dir = os.path.join(params["data_root"], params["dataset_id"])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    feature_map = FeatureMap(params["dataset_id"], data_dir)
    feature_map.load(feature_map_json, params)
    return feature_map


def encoded_h5_to_dataframe(feature_map: FeatureMap, h5_path: str) -> pd.DataFrame:
    data_dict = load_h5(h5_path, verbose=0)
    ordered_columns = list(feature_map.features.keys()) + feature_map.labels
    arrays = []
    for column in ordered_columns:
        values = data_dict[column]
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        arrays.append(values)
    matrix = np.hstack(arrays)
    return pd.DataFrame(matrix, columns=ordered_columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/", help="Config directory")
    parser.add_argument("--expid", type=str, required=True, help="Experiment id")
    parser.add_argument("--num_clients", type=int, default=5, help="Number of FL clients")
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of FL rounds")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local epochs per client")
    parser.add_argument("--iid", action="store_true", help="Use IID data partitioning")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Optional encoded .h5 file path for FL partitioning",
    )
    parser.add_argument(
        "--partition_feature",
        type=str,
        default="f_2",
        help="Feature name for non-IID partitioning",
    )
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--deterministic", action="store_true")
    args = vars(parser.parse_args())

    params = load_config(args["config"], args["expid"])
    params["gpu"] = -1
    set_logger(params)
    logging.info("Federated params: %s", print_to_json(params))

    if args["deterministic"]:
        seed_everything(seed=params["seed"])

    feature_map = ensure_feature_map_and_h5(params)
    ordered_columns = list(feature_map.features.keys()) + feature_map.labels

    data_path = args["data_path"]
    if data_path is None:
        data_path = params["train_data"]
    elif not data_path.endswith(".h5"):
        logging.warning(
            "Ignoring provided data_path=%s because FL requires encoded .h5 for index-safe embeddings.",
            data_path,
        )
        data_path = params["train_data"]

    df = encoded_h5_to_dataframe(feature_map, data_path)

    missing_columns = [col for col in ordered_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Encoded data is missing required columns. Missing: "
            + ", ".join(missing_columns[:20])
        )

    client_frames = partition_data(
        df=df,
        num_clients=args["num_clients"],
        iid=args["iid"],
        partition_feature=args["partition_feature"],
        test_size=0.2,
        val_size=0.1,
        seed=params["seed"],
    )

    model_class = getattr(model_zoo, params["model"])
    server_model = model_class(feature_map, **params)

    strategy = create_server(
        model=server_model,
        num_rounds=args["num_rounds"],
        min_fit_clients=max(1, args["num_clients"] // 2),
        min_eval_clients=max(1, args["num_clients"] // 2),
        min_available_clients=max(1, args["num_clients"]),
        local_epochs=args["local_epochs"],
        batch_size=params.get("batch_size", 1024),
        log_dir="logs",
    )

    server_thread = threading.Thread(
        target=start_server,
        kwargs={
            "strategy": strategy,
            "num_rounds": args["num_rounds"],
            "server_address": args["server_address"],
        },
        daemon=True,
    )
    server_thread.start()

    time.sleep(2)

    with ThreadPoolExecutor(max_workers=args["num_clients"]) as executor:
        futures = []
        for client_id, (train_frame, val_frame) in enumerate(client_frames):
            client_model = model_class(feature_map, **params)
            train_loader = frame_to_dataloader(
                frame=train_frame,
                ordered_columns=ordered_columns,
                batch_size=params.get("batch_size", 1024),
                shuffle=True,
            )
            val_loader = frame_to_dataloader(
                frame=val_frame,
                ordered_columns=ordered_columns,
                batch_size=params.get("batch_size", 1024),
                shuffle=False,
            )

            futures.append(
                executor.submit(
                    start_client,
                    model=client_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    client_id=client_id,
                    server_address=args["server_address"],
                    local_epochs=args["local_epochs"],
                    log_dir="logs",
                )
            )

        for future in futures:
            future.result()

    server_thread.join()


if __name__ == "__main__":
    main()
