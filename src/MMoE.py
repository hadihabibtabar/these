import itertools
from typing import List, Optional, Sequence

import torch
from torch import nn

from fuxictr.pytorch.layers import FeatureEmbedding
from fuxictr.pytorch.models import MultiTaskModel

from .model_components import (
    MLPExpert,
    TaskGate,
    TaskTower,
    TransformerExpert,
    summarize_gates,
)


class MMoELayer(nn.Module):
    """Multi-gate mixture-of-experts layer with task-specific routing."""

    def __init__(
        self,
        num_experts: int,
        num_tasks: int,
        experts: nn.ModuleList,
        gate_input_dim: int,
        gate_hidden_dim: int,
        gate_dropout: float,
        gate_input: str = "mean",
    ) -> None:
        super().__init__()
        if gate_input not in {"mean", "flatten"}:
            raise ValueError("gate_input must be one of {'mean', 'flatten'}")
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.gate_input = gate_input
        self.experts = experts
        self.gates = nn.ModuleList(
            [
                TaskGate(
                    input_dim=gate_input_dim,
                    num_experts=num_experts,
                    hidden_dim=gate_hidden_dim,
                    dropout=gate_dropout,
                )
                for _ in range(num_tasks)
            ]
        )
        self.latest_gate_probabilities: Optional[torch.Tensor] = None

    def forward(self, feature_emb: torch.Tensor):
        # expert_outputs: [B, E, D]
        expert_outputs = torch.stack([expert(feature_emb) for expert in self.experts], dim=1)

        if self.gate_input == "flatten":
            gate_inputs = feature_emb.flatten(start_dim=1)
        else:
            gate_inputs = feature_emb.mean(dim=1)

        task_representations: List[torch.Tensor] = []
        gate_probabilities = []
        for gate in self.gates:
            probs, _ = gate(gate_inputs)  # [B, E]
            task_rep = torch.einsum("be,bed->bd", probs, expert_outputs)
            task_representations.append(task_rep)
            gate_probabilities.append(probs)

        self.latest_gate_probabilities = torch.stack(gate_probabilities, dim=0)  # [T, B, E]
        return task_representations, expert_outputs


class MMoE(MultiTaskModel):
    """Transformer-enhanced MMoE model with research-focused regularization hooks."""

    def __init__(
        self,
        feature_map,
        task=["binary_classification"],
        num_tasks=1,
        model_id="MMoE_Personalized",
        gpu=-1,
        learning_rate=1e-3,
        embedding_dim=32,
        num_experts=8,
        cvr_weight=1.0,
        expert_type="transformer",
        expert_output_dim=None,
        expert_hidden_units=(512, 256),
        gate_hidden_units=(128, 64),
        gate_input="mean",
        gate_dropout=0.1,
        tower_hidden_units=(128, 64),
        tower_dropout=0.1,
        transformer_layers=2,
        transformer_heads=4,
        transformer_dropout=0.1,
        transformer_dim_feedforward=256,
        gate_entropy_weight=0.0,
        gate_balance_weight=0.0,
        tower_orthogonality_weight=0.0,
        uncertainty_weighting=False,
        deterministic=False,
        **kwargs,
    ):
        super().__init__(
            feature_map,
            task=task,
            num_tasks=num_tasks,
            model_id=model_id,
            gpu=gpu,
            **kwargs,
        )

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

        self.num_tasks = num_tasks
        self.gate_entropy_weight = float(gate_entropy_weight)
        self.gate_balance_weight = float(gate_balance_weight)
        self.tower_orthogonality_weight = float(tower_orthogonality_weight)
        self.uncertainty_weighting = bool(uncertainty_weighting)

        # Use net_dropout when provided by legacy configs.
        net_dropout = float(kwargs.get("net_dropout", 0.0))
        if transformer_dropout is None:
            transformer_dropout = net_dropout
        if gate_dropout is None:
            gate_dropout = net_dropout
        if tower_dropout is None:
            tower_dropout = net_dropout

        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.num_fields = feature_map.num_fields

        if expert_output_dim is None:
            expert_output_dim = (
                int(expert_hidden_units[-1]) if len(expert_hidden_units) > 0 else embedding_dim
            )
        self.expert_output_dim = int(expert_output_dim)

        experts = nn.ModuleList(
            [
                self._build_expert(
                    expert_type=expert_type,
                    embedding_dim=embedding_dim,
                    num_fields=self.num_fields,
                    expert_output_dim=self.expert_output_dim,
                    expert_hidden_units=expert_hidden_units,
                    transformer_layers=transformer_layers,
                    transformer_heads=transformer_heads,
                    transformer_dropout=transformer_dropout,
                    transformer_dim_feedforward=transformer_dim_feedforward,
                )
                for _ in range(num_experts)
            ]
        )

        gate_hidden_dim = self._first_unit(gate_hidden_units, fallback=max(32, num_experts * 2))
        gate_input_dim = embedding_dim * self.num_fields if gate_input == "flatten" else embedding_dim
        self.mmoe_layer = MMoELayer(
            num_experts=num_experts,
            num_tasks=self.num_tasks,
            experts=experts,
            gate_input_dim=gate_input_dim,
            gate_hidden_dim=gate_hidden_dim,
            gate_dropout=float(gate_dropout),
            gate_input=gate_input,
        )

        tower_units = self._to_int_list(tower_hidden_units)
        self.towers = nn.ModuleList(
            [
                TaskTower(
                    input_dim=self.expert_output_dim,
                    hidden_units=tower_units,
                    output_dim=1,
                    dropout=float(tower_dropout),
                )
                for _ in range(num_tasks)
            ]
        )

        # Fixed task weights (or trainable uncertainty weights).
        if self.uncertainty_weighting:
            self.log_task_variances = nn.Parameter(torch.zeros(self.num_tasks))
            self.register_buffer("fixed_task_weights", torch.ones(self.num_tasks))
        else:
            task_weights = torch.ones(self.num_tasks)
            if self.num_tasks >= 2:
                task_weights[1] = float(cvr_weight)
            self.register_buffer("fixed_task_weights", task_weights)
            self.log_task_variances = None

        self.latest_gate_diagnostics = None
        self.latest_task_representations: Optional[torch.Tensor] = None

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def _build_expert(
        self,
        expert_type: str,
        embedding_dim: int,
        num_fields: int,
        expert_output_dim: int,
        expert_hidden_units: Sequence[int],
        transformer_layers: int,
        transformer_heads: int,
        transformer_dropout: float,
        transformer_dim_feedforward: int,
    ) -> nn.Module:
        if expert_type.lower() == "mlp":
            return MLPExpert(
                embedding_dim=embedding_dim,
                num_fields=num_fields,
                expert_dim=expert_output_dim,
                hidden_units=self._to_int_list(expert_hidden_units),
                dropout=float(transformer_dropout),
            )
        return TransformerExpert(
            embedding_dim=embedding_dim,
            num_fields=num_fields,
            expert_dim=expert_output_dim,
            num_layers=int(transformer_layers),
            num_heads=int(transformer_heads),
            feedforward_dim=int(transformer_dim_feedforward),
            dropout=float(transformer_dropout),
            norm_first=True,
            use_positional_encoding=True,
            pooling="mean",
        )

    def _first_unit(self, units: Sequence[int], fallback: int) -> int:
        parsed = self._to_int_list(units)
        return parsed[0] if parsed else fallback

    def _to_int_list(self, units: Sequence[int]) -> List[int]:
        if units is None:
            return []
        if isinstance(units, int):
            return [int(units)]
        return [int(u) for u in units]

    def _task_loss_weights(self) -> torch.Tensor:
        weights = self.fixed_task_weights.to(self.device)
        weights = weights / weights.sum() * self.num_tasks
        return weights

    def _orthogonality_penalty(self, task_representations: torch.Tensor) -> torch.Tensor:
        # task_representations: [T, B, D]
        penalties = []
        for i, j in itertools.combinations(range(task_representations.size(0)), 2):
            sim = nn.functional.cosine_similarity(
                task_representations[i], task_representations[j], dim=-1
            )
            penalties.append(sim.abs().mean())
        if not penalties:
            return torch.tensor(0.0, device=task_representations.device)
        return torch.stack(penalties).mean()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)  # [B, F, D]

        task_representations, _ = self.mmoe_layer(feature_emb)
        self.latest_task_representations = torch.stack(task_representations, dim=0)

        gate_probs = self.mmoe_layer.latest_gate_probabilities
        if gate_probs is not None:
            self.latest_gate_diagnostics = summarize_gates(gate_probs)

        labels = self.feature_map.labels
        return_dict = {}
        for task_idx, label_name in enumerate(labels):
            logits, _ = self.towers[task_idx](task_representations[task_idx])
            return_dict[f"{label_name}_pred"] = self.output_activation[task_idx](logits)
        return return_dict

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        labels = self.feature_map.labels

        task_losses = [
            self.loss_fn[i](return_dict[f"{labels[i]}_pred"], y_true[i], reduction="mean")
            for i in range(self.num_tasks)
        ]

        if self.uncertainty_weighting:
            stacked_losses = torch.stack(task_losses)
            precision = torch.exp(-self.log_task_variances)
            total_loss = torch.sum(precision * stacked_losses + self.log_task_variances)
        else:
            weights = self._task_loss_weights()
            total_loss = torch.sum(weights * torch.stack(task_losses))

        if self.latest_gate_diagnostics is not None:
            if self.gate_entropy_weight > 0:
                # Maximize entropy by subtracting it from the minimized objective.
                total_loss = total_loss - self.gate_entropy_weight * self.latest_gate_diagnostics.task_entropy.mean()
            if self.gate_balance_weight > 0:
                total_loss = total_loss + self.gate_balance_weight * self.latest_gate_diagnostics.load_balance_loss

        if (
            self.tower_orthogonality_weight > 0
            and self.latest_task_representations is not None
            and self.num_tasks > 1
        ):
            total_loss = total_loss + self.tower_orthogonality_weight * self._orthogonality_penalty(
                self.latest_task_representations
            )

        return total_loss

    def get_auxiliary_diagnostics(self):
        """Return latest gating diagnostics for monitoring and research logging."""
        diagnostics = {}
        if self.latest_gate_diagnostics is None:
            return diagnostics

        diagnostics["gate_entropy_mean"] = float(self.latest_gate_diagnostics.task_entropy.mean().item())
        diagnostics["gate_balance_loss"] = float(self.latest_gate_diagnostics.load_balance_loss.item())
        for task_idx, entropy in enumerate(self.latest_gate_diagnostics.task_entropy):
            diagnostics[f"task_{task_idx}_gate_entropy"] = float(entropy.item())
        utilization = self.latest_gate_diagnostics.expert_utilization
        for task_idx in range(utilization.size(0)):
            for expert_idx in range(utilization.size(1)):
                diagnostics[f"task_{task_idx}_expert_{expert_idx}_usage"] = float(
                    utilization[task_idx, expert_idx].item()
                )
        return diagnostics

    def estimate_task_gradient_norms(self, inputs):
        """Estimate per-task gradient norms on shared parameters (diagnostics only)."""
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        labels = self.feature_map.labels

        shared_params = [
            param
            for name, param in self.named_parameters()
            if param.requires_grad and ("embedding_layer" in name or "mmoe_layer" in name)
        ]

        grad_norms = []
        for task_idx in range(self.num_tasks):
            loss = self.loss_fn[task_idx](
                return_dict[f"{labels[task_idx]}_pred"], y_true[task_idx], reduction="mean"
            )
            grads = torch.autograd.grad(
                loss,
                shared_params,
                retain_graph=True,
                allow_unused=True,
            )
            squared_norm = torch.tensor(0.0, device=self.device)
            for grad in grads:
                if grad is not None:
                    squared_norm = squared_norm + torch.sum(grad.detach() ** 2)
            grad_norms.append(torch.sqrt(squared_norm + 1e-12))
        return torch.stack(grad_norms)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
