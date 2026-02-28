import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import flwr as fl
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader


class MMoEClient(fl.client.NumPyClient):
    """Flower client wrapping a FuxiCTR-style multitask model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        client_id: int,
        local_epochs: int = 1,
        max_grad_norm: float = 5.0,
        log_dir: str = "logs",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.max_grad_norm = max_grad_norm

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(f"fl_client_{client_id}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_dir / f"client_{client_id}.log")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def get_parameters(self, config):
        return [value.detach().cpu().numpy() for _, value in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        local_epochs = int(config.get("local_epochs", config.get("epochs", self.local_epochs)))

        self.model.train()
        total_examples = 0
        total_loss = 0.0

        for epoch in range(local_epochs):
            for batch in self.train_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                if torch.is_tensor(batch):
                    batch = batch.to(self.model.device)

                self.model.optimizer.zero_grad()
                loss = self.model.get_total_loss(batch)
                loss.backward()
                if self.max_grad_norm and self.max_grad_norm > 0:
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.model.optimizer.step()

                total_loss += float(loss.item())
                total_examples += int(batch.size(0))

            self.logger.info(
                "Client %s local epoch %s/%s finished.",
                self.client_id,
                epoch + 1,
                local_epochs,
            )

        avg_loss = total_loss / max(1, len(self.train_loader) * local_epochs)
        metrics: Dict[str, float] = {"train_loss": avg_loss}

        if hasattr(self.model, "get_auxiliary_diagnostics"):
            diagnostics = self.model.get_auxiliary_diagnostics()
            for key, value in diagnostics.items():
                metrics[key] = float(value)

        return self.get_parameters({}), total_examples, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        val_metrics = self.model.evaluate(self.val_loader)
        metrics = {k: float(v) for k, v in val_metrics.items()}

        loss = metrics.get("logloss")
        if loss is None:
            logloss_keys = [k for k in metrics.keys() if "logloss" in k.lower()]
            if logloss_keys:
                loss = float(np.mean([metrics[k] for k in logloss_keys]))
            else:
                loss = 0.0

        num_examples = getattr(self.val_loader, "num_samples", None)
        if num_examples is None and hasattr(self.val_loader, "dataset"):
            num_examples = len(self.val_loader.dataset)
        num_examples = int(num_examples or 0)

        return float(loss), num_examples, metrics


def create_client(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    client_id: int,
    local_epochs: int = 1,
    max_grad_norm: float = 5.0,
    log_dir: str = "logs",
) -> MMoEClient:
    return MMoEClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        client_id=client_id,
        local_epochs=local_epochs,
        max_grad_norm=max_grad_norm,
        log_dir=log_dir,
    )


def start_client(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    client_id: int,
    server_address: str = "127.0.0.1:8080",
    local_epochs: int = 1,
    max_grad_norm: float = 5.0,
    log_dir: str = "logs",
):
    client = create_client(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        client_id=client_id,
        local_epochs=local_epochs,
        max_grad_norm=max_grad_norm,
        log_dir=log_dir,
    )
    fl.client.start_numpy_client(server_address=server_address, client=client)
