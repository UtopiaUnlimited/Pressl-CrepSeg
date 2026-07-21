from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


PROTOTYPE_METHOD_NAMES = {
    "class_temporal_prototype",
    "class_temporal_prototype_memory",
    "ctpm",
}


def class_temporal_prototype_enabled(config: dict) -> bool:
    prior_cfg = config.get("prior_injection", {}) or {}
    return bool(prior_cfg.get("enabled", False)) and str(
        prior_cfg.get("method", "")
    ).lower() in PROTOTYPE_METHOD_NAMES


class ClassTemporalPrototypeMemory(nn.Module):
    """Query a fold-safe class-month prototype bank at the final Galileo layer.

    The archive contains three banks: patches from partition A, patches from
    partition B, and all training patches. During training, a sample queries
    the opposite partition to prevent its own labels from entering the memory.
    Validation and test samples query the all-training bank.
    """

    def __init__(
        self,
        archive_path: str | Path,
        vision_dim: int,
        num_classes: int,
        num_months: int,
        train_folds: Iterable[int],
        feature_layer_index: int,
        query_dim: int = 128,
        gate_hidden_dim: int = 128,
        dropout: float = 0.0,
        temperature: float = 0.07,
        initial_gate_bias: float = -1.0,
        initial_strength: float = 0.01,
        learnable_strength: bool = True,
        record_diagnostics: bool = False,
    ) -> None:
        super().__init__()
        self.archive_path = Path(archive_path)
        if not self.archive_path.exists():
            raise FileNotFoundError(
                "Class-temporal prototype archive is missing: "
                f"{self.archive_path}. Run scripts/build_class_temporal_prototypes.py first."
            )
        if query_dim < 1 or gate_hidden_dim < 1:
            raise ValueError("query_dim and gate_hidden_dim must be positive.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")

        with np.load(self.archive_path, allow_pickle=False) as archive:
            required = {"prototypes", "mask", "confidence", "counts", "metadata_json"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(
                    f"Prototype archive {self.archive_path} misses {sorted(missing)}."
                )
            prototypes = archive["prototypes"].astype(np.float32, copy=False)
            mask = archive["mask"].astype(np.bool_, copy=False)
            confidence = archive["confidence"].astype(np.float32, copy=False)
            counts = archive["counts"].astype(np.int64, copy=False)
            metadata = json.loads(str(archive["metadata_json"].item()))

        if prototypes.ndim != 5:
            raise ValueError(
                "Prototype archive must store [bank,class,month,prototype,channel], "
                f"got {prototypes.shape}."
            )
        expected_prefix = (3, int(num_classes), int(num_months))
        if tuple(prototypes.shape[:3]) != expected_prefix:
            raise ValueError(
                "Prototype archive does not match the active PASTIS protocol: "
                f"expected {expected_prefix}, got {tuple(prototypes.shape[:3])}."
            )
        if prototypes.shape[-1] != int(vision_dim):
            raise ValueError(
                f"Prototype channel dimension {prototypes.shape[-1]} does not match "
                f"Galileo feature dimension {vision_dim}."
            )
        expected_token_shape = prototypes.shape[:-1]
        if (
            tuple(mask.shape) != expected_token_shape
            or tuple(confidence.shape) != expected_token_shape
            or tuple(counts.shape) != expected_token_shape
        ):
            raise ValueError("Prototype archive mask, confidence, and count shapes must align.")
        if int(metadata.get("partition_modulus", 2)) != 2:
            raise ValueError("Only the documented two-way cross-fit prototype split is supported.")
        archive_layer = int(metadata.get("feature_layer_index", feature_layer_index))
        if archive_layer != int(feature_layer_index):
            raise ValueError(
                "Prototype archive feature layer does not match the config: "
                f"archive={archive_layer}, config={feature_layer_index}."
            )

        # The archive is external run data, not checkpoint state. Keeping it
        # non-persistent avoids duplicating it in every model checkpoint.
        self.register_buffer("prototype_banks", torch.from_numpy(prototypes), persistent=False)
        self.register_buffer("prototype_mask", torch.from_numpy(mask), persistent=False)
        self.register_buffer(
            "prototype_confidence", torch.from_numpy(confidence), persistent=False
        )
        self.register_buffer("prototype_counts", torch.from_numpy(counts), persistent=False)

        self.vision_dim = int(vision_dim)
        self.num_classes = int(num_classes)
        self.num_months = int(num_months)
        self.prototypes_per_group = int(prototypes.shape[3])
        self.feature_layer_index = int(feature_layer_index)
        self.train_folds = frozenset(int(fold) for fold in train_folds)
        self.temperature = float(temperature)
        self.record_diagnostics = bool(record_diagnostics)
        self._latest_diagnostics: dict[str, torch.Tensor] = {}

        self.vision_norm = nn.LayerNorm(self.vision_dim)
        self.query_projection = nn.Linear(self.vision_dim, int(query_dim), bias=False)
        self.key_projection = nn.Linear(self.vision_dim, int(query_dim), bias=False)
        self.value_projection = nn.Linear(self.vision_dim, int(query_dim), bias=False)
        self.output_projection = nn.Linear(int(query_dim), self.vision_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(self.vision_dim + int(query_dim), int(gate_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(gate_hidden_dim), 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, float(initial_gate_bias))

        raw_strength = torch.tensor(float(initial_strength), dtype=torch.float32)
        if learnable_strength:
            self.raw_strength = nn.Parameter(raw_strength)
        else:
            self.register_buffer("raw_strength", raw_strength)

    def _bank_indices(self, batch: dict, batch_size: int, device: torch.device) -> torch.Tensor:
        patch_ids = batch.get("patch_id")
        folds = batch.get("fold")
        if patch_ids is None or folds is None:
            raise ValueError("Prototype memory needs batch['patch_id'] and batch['fold'].")
        if len(patch_ids) != batch_size:
            raise ValueError("batch['patch_id'] must align with the feature batch.")
        if isinstance(folds, torch.Tensor):
            fold_values = [int(value) for value in folds.detach().cpu().tolist()]
        else:
            fold_values = [int(value) for value in folds]
        if len(fold_values) != batch_size:
            raise ValueError("batch['fold'] must align with the feature batch.")

        selected: list[int] = []
        for patch_id, fold in zip(patch_ids, fold_values):
            if fold in self.train_folds:
                # A samples read B's bank and vice versa. Index 2 is the full
                # train-fold bank reserved for validation and test.
                selected.append(1 if int(patch_id) % 2 == 0 else 0)
            else:
                selected.append(2)
        return torch.tensor(selected, dtype=torch.long, device=device)

    def _select_monthly_prototypes(
        self,
        bank_indices: torch.Tensor,
        months: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select the matching class bank for each sample-month."""

        selected_prototypes = self.prototype_banks.index_select(0, bank_indices)
        selected_mask = self.prototype_mask.index_select(0, bank_indices)
        selected_confidence = self.prototype_confidence.index_select(0, bank_indices)

        # [B,C,M,K,D] -> [B,M,C,K,D], then calendar-month lookup.
        by_month = selected_prototypes.permute(0, 2, 1, 3, 4)
        mask_by_month = selected_mask.permute(0, 2, 1, 3)
        confidence_by_month = selected_confidence.permute(0, 2, 1, 3)
        batch_index = torch.arange(months.shape[0], device=months.device)[:, None]
        return (
            by_month[batch_index, months],
            mask_by_month[batch_index, months],
            confidence_by_month[batch_index, months],
        )

    @staticmethod
    def _normalized_entropy(attention: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=-1)
        valid_count = mask.sum(dim=-1).to(attention.dtype)
        maximum = valid_count.clamp_min(2.0).log().unsqueeze(-1)
        normalized = torch.where(
            valid_count.unsqueeze(-1) > 1.0,
            entropy / maximum,
            torch.zeros_like(entropy),
        )
        return normalized.mean()

    def _record(
        self,
        tokens: torch.Tensor,
        residual: torch.Tensor,
        attention: torch.Tensor,
        gate: torch.Tensor,
        mask: torch.Tensor,
        bank_indices: torch.Tensor,
    ) -> None:
        if not self.record_diagnostics:
            return
        with torch.no_grad():
            residual_norm = torch.linalg.vector_norm(
                residual.detach().float().reshape(residual.shape[0], -1), dim=1
            )
            token_norm = torch.linalg.vector_norm(
                tokens.detach().float().reshape(tokens.shape[0], -1), dim=1
            ).clamp_min(1e-8)
            candidate_ratio = (residual_norm / token_norm).mean()
            strength = torch.tanh(self.raw_strength.detach().float())
            self._latest_diagnostics = {
                "strength": strength,
                "gate_mean": gate.detach().float().mean(),
                "gate_std": gate.detach().float().std(unbiased=False),
                "attention_entropy": self._normalized_entropy(
                    attention.detach().float(), mask.detach()
                ),
                "attention_top1": attention.detach().float().max(dim=-1).values.mean(),
                "valid_prototype_fraction": mask.detach().float().mean(),
                "candidate_residual_ratio": candidate_ratio,
                "applied_residual_ratio": strength.abs() * candidate_ratio,
                "crossfit_train_fraction": (bank_indices != 2).float().mean(),
            }

    def pop_prior_diagnostics(self) -> dict[str, torch.Tensor]:
        diagnostics = self._latest_diagnostics
        self._latest_diagnostics = {}
        return diagnostics

    def forward(
        self,
        features: tuple[torch.Tensor, ...] | list[torch.Tensor],
        months: torch.Tensor,
        batch: dict,
        layer_indices: tuple[int, ...] | list[int],
    ) -> tuple[torch.Tensor, ...]:
        features = tuple(features)
        selected_layer_indices = tuple(int(index) for index in layer_indices)
        if self.feature_layer_index not in selected_layer_indices:
            raise ValueError(
                f"Prototype layer {self.feature_layer_index} is not supplied by the decoder: "
                f"{selected_layer_indices}."
            )
        feature_position = selected_layer_indices.index(self.feature_layer_index)
        feature = features[feature_position]
        if feature.ndim != 5:
            raise ValueError(
                f"Prototype memory expects [B,T,D,H,W], got {tuple(feature.shape)}."
            )
        batch_size, timesteps, channels, height, width = feature.shape
        if channels != self.vision_dim:
            raise ValueError(
                f"Expected {self.vision_dim} feature channels, got {channels}."
            )
        if tuple(months.shape) != (batch_size, timesteps):
            raise ValueError(
                "Prototype memory months must be [B,T] aligned with the feature map."
            )
        if months.numel() and (
            int(months.min()) < 0 or int(months.max()) >= self.num_months
        ):
            raise ValueError(
                f"Calendar month indices must be in [0, {self.num_months - 1}]."
            )

        bank_indices = self._bank_indices(batch, batch_size, feature.device)
        prototype, prototype_mask, confidence = self._select_monthly_prototypes(
            bank_indices,
            months,
        )
        prototype = prototype.reshape(
            batch_size,
            timesteps,
            self.num_classes * self.prototypes_per_group,
            channels,
        )
        prototype_mask = prototype_mask.reshape(
            batch_size,
            timesteps,
            self.num_classes * self.prototypes_per_group,
        )
        confidence = confidence.reshape_as(prototype_mask).to(dtype=feature.dtype)
        effective_mask = prototype_mask & (confidence > 0.0)

        tokens = feature.permute(0, 1, 3, 4, 2).reshape(
            batch_size, timesteps, height * width, channels
        )
        vision = self.vision_norm(tokens)
        query = F.normalize(self.query_projection(vision), dim=-1)
        key = F.normalize(self.key_projection(prototype), dim=-1)
        value = self.value_projection(prototype)
        scores = torch.einsum("btnq,btpq->btnp", query, key) / self.temperature
        scores = scores + confidence.clamp_min(1e-8).log().unsqueeze(2)

        token_count = effective_mask.shape[-1]
        no_valid_token = ~effective_mask.any(dim=-1)
        fallback_mask = (
            torch.arange(token_count, device=feature.device).view(1, 1, token_count)
            == 0
        )
        safe_mask = effective_mask | (no_valid_token.unsqueeze(-1) & fallback_mask)
        scores = scores.masked_fill(~safe_mask.unsqueeze(2), float("-inf"))
        attention = torch.softmax(scores.float(), dim=-1).to(dtype=feature.dtype)
        context = torch.einsum("btnp,btpd->btnd", attention, value)
        context = context * (~no_valid_token).to(context.dtype).unsqueeze(-1).unsqueeze(-1)
        residual = self.output_projection(context)
        gate = torch.sigmoid(self.gate(torch.cat([vision, context], dim=-1)))
        residual = gate * residual
        output_tokens = tokens + torch.tanh(self.raw_strength) * residual
        output = (
            output_tokens.reshape(batch_size, timesteps, height, width, channels)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )
        self._record(
            tokens=tokens,
            residual=residual,
            attention=attention,
            gate=gate,
            mask=effective_mask,
            bank_indices=bank_indices,
        )

        enhanced = list(features)
        enhanced[feature_position] = output
        return tuple(enhanced)


def build_class_temporal_prototype_memory(
    config: dict,
    feature_channels: int,
    num_layers: int,
) -> ClassTemporalPrototypeMemory | None:
    if not class_temporal_prototype_enabled(config):
        return None
    prior_cfg = config.get("prior_injection", {}) or {}
    source_cfg = prior_cfg.get("source", {}) or {}
    if not isinstance(source_cfg, dict):
        raise ValueError("Class-temporal prototype source must be a mapping.")
    archive_path = source_cfg.get("path")
    if not archive_path:
        raise ValueError("Class-temporal prototype source needs a prototype archive path.")
    fusion_cfg = prior_cfg.get("fusion", {}) or {}
    diagnostics_cfg = prior_cfg.get("diagnostics", {}) or {}
    requested_layer = int(fusion_cfg.get("feature_layer_index", -1))
    feature_layer_index = requested_layer if requested_layer >= 0 else num_layers + requested_layer
    if feature_layer_index < 0 or feature_layer_index >= num_layers:
        raise ValueError(
            f"feature_layer_index={requested_layer} is invalid for {num_layers} cached layers."
        )
    module = ClassTemporalPrototypeMemory(
        archive_path=archive_path,
        vision_dim=int(feature_channels),
        num_classes=int(config["data"]["num_classes"]),
        num_months=int(config["data"].get("selected_timesteps", 12)),
        train_folds=config["data"].get("train_folds", (1, 2, 3)),
        feature_layer_index=feature_layer_index,
        query_dim=int(fusion_cfg.get("query_dim", 128)),
        gate_hidden_dim=int(fusion_cfg.get("gate_hidden_dim", 128)),
        dropout=float(fusion_cfg.get("dropout", 0.0)),
        temperature=float(fusion_cfg.get("temperature", 0.07)),
        initial_gate_bias=float(fusion_cfg.get("initial_gate_bias", -1.0)),
        initial_strength=float(fusion_cfg.get("initial_strength", 0.01)),
        learnable_strength=bool(fusion_cfg.get("learnable_strength", True)),
        record_diagnostics=bool(diagnostics_cfg.get("enabled", False)),
    )
    expected_k = source_cfg.get("prototypes_per_group")
    if expected_k is not None and int(expected_k) != module.prototypes_per_group:
        raise ValueError(
            "Prototype archive count does not match the config: "
            f"archive={module.prototypes_per_group}, config={expected_k}."
        )
    return module
