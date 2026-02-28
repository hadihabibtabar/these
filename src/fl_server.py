import logging
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import Metrics
from flwr.server.strategy import FedAvg
import torch
from torch import nn


def aggregate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated: Dict[str, float] = {}
    metric_keys = set()
    for _, metric_dict in metrics:
        metric_keys.update(metric_dict.keys())

    for key in metric_keys:
        weighted_sum = 0.0
        weight_denom = 0
        for num_examples, metric_dict in metrics:
            if key in metric_dict:
                weighted_sum += float(metric_dict[key]) * num_examples
                weight_denom += num_examples
        if weight_denom > 0:
            aggregated[key] = weighted_sum / weight_denom
    aggregated["num_examples"] = float(total_examples)
    return aggregated


def build_fit_config(local_epochs: int, batch_size: int):
    def fit_config(round_number: int):
        return {
            "round": int(round_number),
            "local_epochs": int(local_epochs),
            "batch_size": int(batch_size),
        }

    return fit_config


def build_eval_config(batch_size: int):
    def eval_config(round_number: int):
        return {
            "round": int(round_number),
            "batch_size": int(batch_size),
        }

    return eval_config


def create_server(
    model: nn.Module,
    num_rounds: int,
    fraction_fit: float = 1.0,
    fraction_eval: float = 1.0,
    min_fit_clients: int = 2,
    min_eval_clients: int = 2,
    min_available_clients: int = 2,
    local_epochs: int = 1,
    batch_size: int = 1024,
    log_dir: str = "logs",
) -> FedAvg:
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    logger = logging.getLogger("fl_server")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_path / "server.log")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)

    initial_parameters = fl.common.ndarrays_to_parameters(
        [value.detach().cpu().numpy() for _, value in model.state_dict().items()]
    )

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_eval,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_eval_clients,
        min_available_clients=min_available_clients,
        on_fit_config_fn=build_fit_config(local_epochs=local_epochs, batch_size=batch_size),
        on_evaluate_config_fn=build_eval_config(batch_size=batch_size),
        evaluate_metrics_aggregation_fn=aggregate_metrics,
        initial_parameters=initial_parameters,
    )
    logger.info(
        "Initialized FedAvg strategy for %s rounds, min clients fit/eval/available = %s/%s/%s",
        num_rounds,
        min_fit_clients,
        min_eval_clients,
        min_available_clients,
    )
    return strategy


def start_server(
    strategy: FedAvg,
    num_rounds: int,
    server_address: str = "127.0.0.1:8080",
):
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
