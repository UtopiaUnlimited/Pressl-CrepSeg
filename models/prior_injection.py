from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PriorBatch:
    """One batch of encoded heterogeneous prior tokens.

    The fusion module only depends on this representation. Task-specific
    adapters are responsible for converting tables, metadata, or text into
    these tensors.
    """

    tokens: torch.Tensor
    mask: torch.Tensor
    confidence: torch.Tensor
    type_ids: torch.Tensor
    entity_ids: torch.Tensor | None = None
    time_values: torch.Tensor | None = None
    source_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3:
            raise ValueError(
                f"prior tokens must be [B,N,D], got {tuple(self.tokens.shape)}"
            )
        batch, num_tokens, token_dim = self.tokens.shape
        if batch < 1 or num_tokens < 1 or token_dim < 1:
            raise ValueError(
                "prior tokens need positive batch, token, and channel dimensions."
            )
        expected = (batch, num_tokens)
        for name, value in (
            ("mask", self.mask),
            ("confidence", self.confidence),
            ("type_ids", self.type_ids),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"prior {name} must be {expected}, got {tuple(value.shape)}"
                )
        if self.mask.dtype != torch.bool:
            raise TypeError("prior mask must use torch.bool.")
        if not self.confidence.is_floating_point():
            raise TypeError("prior confidence must be floating point.")
        if self.type_ids.dtype != torch.long:
            raise TypeError("prior type_ids must use torch.long.")
        if self.source_names is not None and not self.source_names:
            raise ValueError("prior source_names must be non-empty when provided.")
        if self.entity_ids is not None:
            if tuple(self.entity_ids.shape) != expected:
                raise ValueError(
                    "prior entity_ids must match [B,N], "
                    f"got {tuple(self.entity_ids.shape)}"
                )
            if self.entity_ids.dtype != torch.long:
                raise TypeError("prior entity_ids must use torch.long.")
        if self.time_values is not None:
            if (
                self.time_values.ndim != 3
                or tuple(self.time_values.shape[:2]) != expected
            ):
                raise ValueError(
                    "prior time_values must be [B,N,K], "
                    f"got {tuple(self.time_values.shape)}"
                )
            if not self.time_values.is_floating_point():
                raise TypeError("prior time_values must be floating point.")


class PriorTokenEncoder(nn.Module):
    """Common task-adapter boundary for heterogeneous prior sources."""

    def forward(
        self,
        batch_size: int,
        batch: dict | None = None,
    ) -> PriorBatch:
        raise NotImplementedError


class StructuredPriorEncoder(nn.Module):
    """Encode numeric, categorical, cyclic-time, and confidence fields.

    This encoder is task agnostic. A task adapter supplies aligned tensors and
    receives a :class:`PriorBatch` with one common token dimension.
    """

    def __init__(
        self,
        numeric_dim: int,
        token_dim: int,
        hidden_dim: int,
        num_types: int,
        num_entities: int | None = None,
        time_frequencies: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if numeric_dim < 1 or token_dim < 1 or hidden_dim < 1:
            raise ValueError("numeric_dim, token_dim, and hidden_dim must be positive.")
        if num_types < 1 or time_frequencies < 1:
            raise ValueError("num_types and time_frequencies must be positive.")
        if num_entities is not None and int(num_entities) < 1:
            raise ValueError("num_entities must be positive when provided.")

        self.numeric_dim = int(numeric_dim)
        self.token_dim = int(token_dim)
        self.num_types = int(num_types)
        self.num_entities = None if num_entities is None else int(num_entities)
        self.time_frequencies = int(time_frequencies)

        self.numeric_projection = nn.Sequential(
            nn.Linear(self.numeric_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.token_dim),
        )
        self.confidence_projection = nn.Sequential(
            nn.Linear(1, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.token_dim),
        )
        self.time_projection = nn.Linear(
            self.time_frequencies * 2,
            self.token_dim,
            bias=False,
        )
        self.type_embedding = nn.Embedding(self.num_types, self.token_dim)
        self.entity_embedding = (
            nn.Embedding(self.num_entities, self.token_dim)
            if self.num_entities is not None
            else None
        )
        self.output_norm = nn.LayerNorm(self.token_dim)
        self.output_dropout = nn.Dropout(float(dropout))
        self.register_buffer(
            "frequencies",
            torch.pow(2.0, torch.arange(self.time_frequencies, dtype=torch.float32)),
            persistent=False,
        )

        nn.init.trunc_normal_(self.type_embedding.weight, std=0.02)
        if self.entity_embedding is not None:
            nn.init.trunc_normal_(self.entity_embedding.weight, std=0.02)

    def _encode_time(self, time_values: torch.Tensor) -> torch.Tensor:
        if time_values.ndim == 3 and time_values.shape[-1] == 1:
            time_values = time_values.squeeze(-1)
        if time_values.ndim != 2:
            raise ValueError(
                "cyclic time_values must be [B,N] or [B,N,1], "
                f"got {tuple(time_values.shape)}"
            )
        angles = (
            2.0
            * math.pi
            * time_values.unsqueeze(-1)
            * self.frequencies.to(device=time_values.device, dtype=time_values.dtype)
        )
        return self.time_projection(torch.cat((angles.sin(), angles.cos()), dim=-1))

    def forward(
        self,
        numeric_values: torch.Tensor,
        mask: torch.Tensor,
        confidence: torch.Tensor,
        type_ids: torch.Tensor,
        entity_ids: torch.Tensor | None = None,
        time_values: torch.Tensor | None = None,
    ) -> PriorBatch:
        if numeric_values.ndim != 3 or numeric_values.shape[-1] != self.numeric_dim:
            raise ValueError(
                f"numeric_values must be [B,N,{self.numeric_dim}], "
                f"got {tuple(numeric_values.shape)}"
            )
        batch, num_tokens, _ = numeric_values.shape
        expected = (batch, num_tokens)
        if tuple(mask.shape) != expected or mask.dtype != torch.bool:
            raise ValueError("mask must be bool [B,N].")
        if tuple(confidence.shape) != expected or not confidence.is_floating_point():
            raise ValueError("confidence must be floating-point [B,N].")
        if tuple(type_ids.shape) != expected or type_ids.dtype != torch.long:
            raise ValueError("type_ids must be torch.long [B,N].")
        tokens = self.numeric_projection(numeric_values)
        tokens = tokens + self.confidence_projection(confidence.unsqueeze(-1))
        tokens = tokens + self.type_embedding(type_ids)

        if self.entity_embedding is not None:
            if entity_ids is None:
                raise ValueError(
                    "entity_ids are required by this structured prior encoder."
                )
            if tuple(entity_ids.shape) != expected or entity_ids.dtype != torch.long:
                raise ValueError("entity_ids must be torch.long [B,N].")
            tokens = tokens + self.entity_embedding(entity_ids)
        elif entity_ids is not None:
            raise ValueError(
                "entity_ids were provided but num_entities is not configured."
            )

        stored_time = None
        if time_values is not None:
            if tuple(time_values.shape[:2]) != expected:
                raise ValueError("time_values must start with [B,N].")
            tokens = tokens + self._encode_time(time_values)
            stored_time = (
                time_values.unsqueeze(-1) if time_values.ndim == 2 else time_values
            )

        tokens = self.output_dropout(self.output_norm(tokens))
        tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        return PriorBatch(
            tokens=tokens,
            mask=mask,
            confidence=confidence,
            type_ids=type_ids,
            entity_ids=entity_ids,
            time_values=stored_time,
        )


@dataclass(frozen=True)
class PriorFusionDiagnostics:
    attention: torch.Tensor
    gate: torch.Tensor
    residual: torch.Tensor
    source_weights: torch.Tensor | None = None
    channel_scale: torch.Tensor | None = None
    channel_shift: torch.Tensor | None = None


class ContentAwarePriorFusion(nn.Module):
    """Let visual tokens query a masked, confidence-weighted prior set."""

    def __init__(
        self,
        vision_dim: int,
        prior_dim: int,
        attention_dim: int,
        num_heads: int,
        gate_hidden_dim: int,
        dropout: float = 0.0,
        confidence_bias_scale: float = 1.0,
        source_balance_bias_scale: float = 0.0,
        initial_gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if min(vision_dim, prior_dim, attention_dim, num_heads, gate_hidden_dim) < 1:
            raise ValueError("all CA-HPI dimensions must be positive.")
        if attention_dim % num_heads:
            raise ValueError("attention_dim must be divisible by num_heads.")
        if not math.isfinite(float(confidence_bias_scale)):
            raise ValueError("confidence_bias_scale must be finite.")
        if (
            not math.isfinite(float(source_balance_bias_scale))
            or float(source_balance_bias_scale) < 0.0
        ):
            raise ValueError(
                "source_balance_bias_scale must be finite and non-negative."
            )
        if not math.isfinite(float(initial_gate_bias)):
            raise ValueError("initial_gate_bias must be finite.")

        self.vision_dim = int(vision_dim)
        self.prior_dim = int(prior_dim)
        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attention_dim // self.num_heads
        self.confidence_bias_scale = float(confidence_bias_scale)
        self.source_balance_bias_scale = float(source_balance_bias_scale)

        self.vision_norm = nn.LayerNorm(self.vision_dim)
        self.prior_norm = nn.LayerNorm(self.prior_dim)
        self.query_projection = nn.Linear(self.vision_dim, self.attention_dim)
        self.key_projection = nn.Linear(self.prior_dim, self.attention_dim)
        self.value_projection = nn.Linear(self.prior_dim, self.attention_dim)
        self.output_projection = nn.Linear(
            self.attention_dim,
            self.vision_dim,
            bias=False,
        )
        self.attention_dropout = nn.Dropout(float(dropout))
        self.gate = nn.Sequential(
            nn.Linear(self.vision_dim * 2, int(gate_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(gate_hidden_dim), 1),
        )
        nn.init.constant_(self.gate[-1].bias, float(initial_gate_bias))

    def _attention_components(
        self,
        vision_tokens: torch.Tensor,
        prior: PriorBatch,
        query_bias: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Project inputs and return normalized vision, attention, values, and mask."""

        if vision_tokens.ndim != 3 or vision_tokens.shape[-1] != self.vision_dim:
            raise ValueError(
                f"vision_tokens must be [B,N,{self.vision_dim}], "
                f"got {tuple(vision_tokens.shape)}"
            )
        if prior.tokens.shape[0] != vision_tokens.shape[0]:
            raise ValueError("vision and prior batch sizes do not match.")
        if prior.tokens.shape[-1] != self.prior_dim:
            raise ValueError(
                f"expected prior token dim {self.prior_dim}, "
                f"got {prior.tokens.shape[-1]}"
            )
        if prior.tokens.device != vision_tokens.device:
            raise ValueError("vision and prior tokens must be on the same device.")

        normalized_vision = self.vision_norm(vision_tokens)
        if query_bias is not None:
            if tuple(query_bias.shape) != (self.vision_dim,):
                raise ValueError(
                    f"query_bias must be [{self.vision_dim}], got {tuple(query_bias.shape)}"
                )
            normalized_vision = normalized_vision + query_bias.view(1, 1, -1)
        normalized_prior = self.prior_norm(prior.tokens)

        batch, num_visual, _ = vision_tokens.shape
        num_prior = prior.tokens.shape[1]
        query = (
            self.query_projection(normalized_vision)
            .view(batch, num_visual, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.key_projection(normalized_prior)
            .view(batch, num_prior, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.value_projection(normalized_prior)
            .view(batch, num_prior, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        confidence = prior.confidence.to(device=scores.device, dtype=scores.dtype)
        effective_mask = prior.mask.to(device=scores.device) & (confidence > 0.0)
        confidence_bias = torch.log(confidence.clamp_min(1e-6))
        scores = scores + self.confidence_bias_scale * confidence_bias[:, None, None, :]
        if self.source_balance_bias_scale > 0.0:
            type_ids = prior.type_ids.to(device=scores.device)
            source_counts = torch.zeros(
                (batch, num_prior),
                device=scores.device,
                dtype=scores.dtype,
            )
            source_counts.scatter_add_(
                1,
                type_ids,
                effective_mask.to(dtype=scores.dtype),
            )
            token_source_counts = source_counts.gather(1, type_ids).clamp_min(1.0)
            source_balance_bias = -torch.log(token_source_counts)
            scores = scores + (
                self.source_balance_bias_scale * source_balance_bias[:, None, None, :]
            )
        scores = scores.masked_fill(
            ~effective_mask[:, None, None, :],
            torch.finfo(scores.dtype).min,
        )

        attention = torch.softmax(scores, dim=-1)
        attention = attention * effective_mask[:, None, None, :].to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return normalized_vision, attention, value, effective_mask

    def forward(
        self,
        vision_tokens: torch.Tensor,
        prior: PriorBatch,
        query_bias: torch.Tensor | None = None,
    ) -> PriorFusionDiagnostics:
        normalized_vision, attention, value, _ = self._attention_components(
            vision_tokens,
            prior,
            query_bias=query_bias,
        )
        dropped_attention = self.attention_dropout(attention)

        batch, num_visual, _ = vision_tokens.shape
        context = torch.matmul(dropped_attention, value)
        context = context.transpose(1, 2).reshape(batch, num_visual, self.attention_dim)
        delta = self.output_projection(context)
        gate = torch.sigmoid(self.gate(torch.cat((normalized_vision, delta), dim=-1)))
        residual = gate * delta
        return PriorFusionDiagnostics(
            attention=attention,
            gate=gate,
            residual=residual,
        )


class SourceAwareSpatialFiLMFusion(ContentAwarePriorFusion):
    """Fuse heterogeneous priors with source hierarchy and spatial-channel FiLM.

    Visual queries first select tokens *within* every available prior source.
    A shared content-aware source gate then combines the source contexts without
    allowing a large token library to suppress smaller metadata sources.  The
    resulting context produces channel-wise FiLM parameters, while a spatial
    gate decides where the modulation should enter the decoder input.
    """

    def __init__(
        self,
        vision_dim: int,
        prior_dim: int,
        attention_dim: int,
        num_heads: int,
        gate_hidden_dim: int,
        source_gate_hidden_dim: int,
        film_hidden_dim: int,
        dropout: float = 0.0,
        confidence_bias_scale: float = 1.0,
        source_balance_bias_scale: float = 1.0,
        initial_gate_bias: float = -1.0,
    ) -> None:
        super().__init__(
            vision_dim=vision_dim,
            prior_dim=prior_dim,
            attention_dim=attention_dim,
            num_heads=num_heads,
            gate_hidden_dim=gate_hidden_dim,
            dropout=dropout,
            confidence_bias_scale=confidence_bias_scale,
            source_balance_bias_scale=source_balance_bias_scale,
            initial_gate_bias=initial_gate_bias,
        )
        if min(int(source_gate_hidden_dim), int(film_hidden_dim)) < 1:
            raise ValueError(
                "source_gate_hidden_dim and film_hidden_dim must be positive."
            )

        self.source_gate = nn.Sequential(
            nn.Linear(
                self.vision_dim + self.attention_dim, int(source_gate_hidden_dim)
            ),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(source_gate_hidden_dim), 1),
        )
        self.film = nn.Sequential(
            nn.LayerNorm(self.vision_dim),
            nn.Linear(self.vision_dim, int(film_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(film_hidden_dim), self.vision_dim * 2),
        )

    @staticmethod
    def _num_sources(prior: PriorBatch) -> int:
        if prior.source_names is not None:
            return len(prior.source_names)
        if bool((prior.type_ids < 0).any()):
            raise ValueError("prior type_ids must be non-negative source identifiers.")
        return int(prior.type_ids.max().item()) + 1

    def forward(
        self,
        vision_tokens: torch.Tensor,
        prior: PriorBatch,
        query_bias: torch.Tensor | None = None,
    ) -> PriorFusionDiagnostics:
        normalized_vision, attention, value, effective_mask = (
            self._attention_components(
                vision_tokens,
                prior,
                query_bias=query_bias,
            )
        )
        batch, num_visual, _ = vision_tokens.shape
        num_sources = self._num_sources(prior)
        type_ids = prior.type_ids.to(device=attention.device)
        if bool((type_ids >= num_sources).any()):
            raise ValueError("prior type_ids exceed the declared source_names.")

        source_contexts: list[torch.Tensor] = []
        source_evidence: list[torch.Tensor] = []
        source_validity: list[torch.Tensor] = []
        for source_id in range(num_sources):
            source_mask = effective_mask & (type_ids == source_id)
            source_valid = source_mask.any(dim=-1)
            masked_attention = attention * source_mask[:, None, None, :].to(
                attention.dtype
            )
            source_mass = masked_attention.sum(dim=-1, keepdim=True)
            within_source_attention = masked_attention / source_mass.clamp_min(1e-6)
            within_source_attention = self.attention_dropout(within_source_attention)
            source_context = torch.matmul(within_source_attention, value)
            source_context = source_context.transpose(1, 2).reshape(
                batch,
                num_visual,
                self.attention_dim,
            )
            source_contexts.append(source_context)
            source_evidence.append(
                source_mass.squeeze(-1).mean(dim=1).clamp_min(1e-6).log()
            )
            source_validity.append(source_valid)

        stacked_contexts = torch.stack(source_contexts, dim=2)
        stacked_evidence = torch.stack(source_evidence, dim=-1)
        valid_sources = torch.stack(source_validity, dim=-1)
        expanded_vision = normalized_vision.unsqueeze(2).expand(
            -1,
            -1,
            num_sources,
            -1,
        )
        source_logits = self.source_gate(
            torch.cat((expanded_vision, stacked_contexts), dim=-1)
        ).squeeze(-1)
        source_logits = source_logits + stacked_evidence
        source_logits = source_logits.masked_fill(
            ~valid_sources[:, None, :],
            torch.finfo(source_logits.dtype).min,
        )
        source_weights = torch.softmax(source_logits, dim=-1)
        source_weights = source_weights * valid_sources[:, None, :].to(
            source_weights.dtype
        )
        source_weights = source_weights / source_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

        context = (source_weights.unsqueeze(-1) * stacked_contexts).sum(dim=2)
        context_delta = self.output_projection(context)
        channel_scale, channel_shift = self.film(context_delta).chunk(2, dim=-1)
        channel_scale = torch.tanh(channel_scale)
        channel_shift = torch.tanh(channel_shift)
        gate = torch.sigmoid(
            self.gate(torch.cat((normalized_vision, context_delta), dim=-1))
        )
        has_valid_prior = effective_mask.any(dim=-1)[:, None, None].to(gate.dtype)
        residual = (
            gate * (channel_scale * normalized_vision + channel_shift) * has_valid_prior
        )
        return PriorFusionDiagnostics(
            attention=attention,
            gate=gate,
            residual=residual,
            source_weights=source_weights,
            channel_scale=channel_scale,
            channel_shift=channel_shift,
        )


class TemporalFeaturePyramidPriorInjection(nn.Module):
    """Apply one shared heterogeneous-prior block before a temporal decoder.

    Each feature uses ``[B,T,D,H,W]``. Fusion weights are shared across
    encoder layers; layer embeddings and zero-initialized residual strengths
    allow layer-specific behavior without coupling the module to a decoder.
    """

    def __init__(
        self,
        vision_dim: int,
        prior_dim: int,
        num_layers: int,
        attention_dim: int,
        num_heads: int,
        gate_hidden_dim: int,
        fusion_mode: str = "attention_residual",
        source_gate_hidden_dim: int | None = None,
        film_hidden_dim: int | None = None,
        dropout: float = 0.0,
        confidence_bias_scale: float = 1.0,
        source_balance_bias_scale: float = 0.0,
        initial_gate_bias: float = -2.0,
        initial_strength: float = 0.0,
        learnable_strength: bool = True,
        record_diagnostics: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive.")
        if not math.isfinite(float(initial_strength)):
            raise ValueError("initial_strength must be finite.")

        self.vision_dim = int(vision_dim)
        self.num_layers = int(num_layers)
        self.record_diagnostics = bool(record_diagnostics)
        self._latest_diagnostics: dict[str, torch.Tensor] = {}
        normalized_fusion_mode = str(fusion_mode).lower()
        if normalized_fusion_mode in {"attention_residual", "ca_hpi"}:
            self.fusion = ContentAwarePriorFusion(
                vision_dim=self.vision_dim,
                prior_dim=int(prior_dim),
                attention_dim=int(attention_dim),
                num_heads=int(num_heads),
                gate_hidden_dim=int(gate_hidden_dim),
                dropout=float(dropout),
                confidence_bias_scale=float(confidence_bias_scale),
                source_balance_bias_scale=float(source_balance_bias_scale),
                initial_gate_bias=float(initial_gate_bias),
            )
        elif normalized_fusion_mode in {
            "source_aware_spatial_film",
            "source_aware_film",
            "spatial_film",
        }:
            self.fusion = SourceAwareSpatialFiLMFusion(
                vision_dim=self.vision_dim,
                prior_dim=int(prior_dim),
                attention_dim=int(attention_dim),
                num_heads=int(num_heads),
                gate_hidden_dim=int(gate_hidden_dim),
                source_gate_hidden_dim=int(source_gate_hidden_dim or gate_hidden_dim),
                film_hidden_dim=int(film_hidden_dim or gate_hidden_dim),
                dropout=float(dropout),
                confidence_bias_scale=float(confidence_bias_scale),
                source_balance_bias_scale=float(source_balance_bias_scale),
                initial_gate_bias=float(initial_gate_bias),
            )
        else:
            raise ValueError(f"Unsupported prior fusion_mode: {fusion_mode}")
        self.fusion_mode = normalized_fusion_mode
        self.layer_embedding = nn.Parameter(
            torch.empty(self.num_layers, self.vision_dim)
        )
        nn.init.trunc_normal_(self.layer_embedding, std=0.02)

        raw_strength = torch.full((self.num_layers,), float(initial_strength))
        if learnable_strength:
            self.raw_strength = nn.Parameter(raw_strength)
        else:
            self.register_buffer("raw_strength", raw_strength)

    @staticmethod
    def _summarize_layer(
        diagnostics: PriorFusionDiagnostics,
        tokens: torch.Tensor,
        prior: PriorBatch,
        raw_strength: torch.Tensor,
        strength: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Reduce one layer to detached scalar diagnostics.

        The full attention and gate tensors are never retained. Ratios are
        averaged per sample so that their scale does not depend on batch size.
        """

        with torch.no_grad():
            attention = diagnostics.attention.detach().float()
            gate = diagnostics.gate.detach().float()
            residual = diagnostics.residual.detach().float()
            vision = tokens.detach().float()

            entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=-1)
            confidence = prior.confidence.detach().to(
                device=attention.device,
                dtype=attention.dtype,
            )
            effective_mask = prior.mask.to(device=attention.device) & (confidence > 0.0)
            valid_count = effective_mask.sum(dim=-1).to(attention.dtype)
            maximum_entropy = valid_count.clamp_min(2.0).log()[:, None, None]
            normalized_entropy = torch.where(
                (valid_count > 1.0)[:, None, None],
                entropy / maximum_entropy,
                torch.zeros_like(entropy),
            )

            residual_norm = torch.linalg.vector_norm(
                residual.reshape(residual.shape[0], -1),
                dim=1,
            )
            vision_norm = torch.linalg.vector_norm(
                vision.reshape(vision.shape[0], -1),
                dim=1,
            ).clamp_min(1e-8)
            candidate_residual_ratio = (residual_norm / vision_norm).mean()

            confidence_mask = effective_mask.to(confidence.dtype)
            valid_confidence_mean = (
                (confidence * confidence_mask).sum(dim=-1) / valid_count.clamp_min(1.0)
            ).mean()
            attended_confidence = (
                (attention * confidence[:, None, None, :]).sum(dim=-1).mean()
            )

            summary = {
                "raw_strength": raw_strength.detach().float(),
                "strength": strength.detach().float(),
                "gate_mean": gate.mean(),
                "gate_std": gate.std(unbiased=False),
                "gate_low_fraction": (gate < 0.05).float().mean(),
                "gate_high_fraction": (gate > 0.95).float().mean(),
                "attention_entropy": normalized_entropy.mean(),
                "attention_top1": attention.max(dim=-1).values.mean(),
                "attended_confidence": attended_confidence,
                "valid_confidence_mean": valid_confidence_mean,
                "valid_prior_fraction": effective_mask.float().mean(),
                "candidate_residual_ratio": candidate_residual_ratio,
                "applied_residual_ratio": (
                    strength.detach().float().abs() * candidate_residual_ratio
                ),
            }
            if diagnostics.channel_scale is not None:
                channel_scale = diagnostics.channel_scale.detach().float()
                summary["film_scale_abs_mean"] = channel_scale.abs().mean()
                summary["film_scale_std"] = channel_scale.std(unbiased=False)
            if diagnostics.channel_shift is not None:
                channel_shift = diagnostics.channel_shift.detach().float()
                summary["film_shift_abs_mean"] = channel_shift.abs().mean()
            if prior.source_names is not None and len(prior.source_names) > 1:
                type_ids = prior.type_ids.detach().to(device=attention.device)
                total_valid = valid_count.sum().clamp_min(1.0)
                for source_id, source_name in enumerate(prior.source_names):
                    source_mask = effective_mask & (type_ids == source_id)
                    if diagnostics.source_weights is None:
                        source_attention_mass = (
                            (
                                attention
                                * source_mask[:, None, None, :].to(attention.dtype)
                            )
                            .sum(dim=-1)
                            .mean()
                        )
                    else:
                        source_attention_mass = (
                            diagnostics.source_weights.detach()
                            .float()[..., source_id]
                            .mean()
                        )
                    diagnostic_name = str(source_name).replace("/", "_")
                    summary[f"{diagnostic_name}/attention_mass"] = source_attention_mass
                    summary[f"{diagnostic_name}/valid_token_fraction"] = (
                        source_mask.sum().to(attention.dtype) / total_valid
                    )
            return summary

    def pop_prior_diagnostics(self) -> dict[str, torch.Tensor]:
        """Return and clear the most recent detached diagnostic snapshot."""

        diagnostics = self._latest_diagnostics
        self._latest_diagnostics = {}
        return diagnostics

    def forward(
        self,
        features: tuple[torch.Tensor, ...] | list[torch.Tensor],
        prior: PriorBatch,
        layer_indices: tuple[int, ...] | list[int] | None = None,
        return_diagnostics: bool = False,
    ) -> (
        tuple[torch.Tensor, ...]
        | tuple[tuple[torch.Tensor, ...], tuple[dict[str, torch.Tensor], ...]]
    ):
        features = tuple(features)
        if self.record_diagnostics:
            self._latest_diagnostics = {}
        if layer_indices is None:
            if len(features) != self.num_layers:
                raise ValueError(
                    f"expected {self.num_layers} temporal feature layers, got {len(features)}"
                )
            selected_layer_indices = tuple(range(self.num_layers))
        else:
            selected_layer_indices = tuple(int(index) for index in layer_indices)
            if len(selected_layer_indices) != len(features):
                raise ValueError("layer_indices must match the supplied feature count.")
            if any(
                index < 0 or index >= self.num_layers
                for index in selected_layer_indices
            ):
                raise ValueError(
                    f"layer_indices must be in [0, {self.num_layers - 1}]."
                )

        enhanced: list[torch.Tensor] = []
        summaries: list[dict[str, torch.Tensor]] = []
        summary_layer_indices: list[int] = []
        for layer_index, feature in zip(selected_layer_indices, features):
            if feature.ndim != 5:
                raise ValueError(
                    f"temporal features must be [B,T,D,H,W], got {tuple(feature.shape)}"
                )
            if feature.shape[2] != self.vision_dim:
                raise ValueError(
                    f"expected {self.vision_dim} feature channels, got {feature.shape[2]}"
                )
            batch, timesteps, channels, height, width = feature.shape
            tokens = feature.permute(0, 1, 3, 4, 2).reshape(
                batch, timesteps * height * width, channels
            )
            diagnostics = self.fusion(
                tokens,
                prior,
                query_bias=self.layer_embedding[layer_index],
            )
            strength = torch.tanh(self.raw_strength[layer_index])
            output_tokens = tokens + strength * diagnostics.residual
            output = (
                output_tokens.reshape(batch, timesteps, height, width, channels)
                .permute(0, 1, 4, 2, 3)
                .contiguous()
            )
            enhanced.append(output)

            if return_diagnostics or self.record_diagnostics:
                summaries.append(
                    self._summarize_layer(
                        diagnostics=diagnostics,
                        tokens=tokens,
                        prior=prior,
                        raw_strength=self.raw_strength[layer_index],
                        strength=strength,
                    )
                )
                summary_layer_indices.append(layer_index)

        if self.record_diagnostics:
            self._latest_diagnostics = {
                f"layer_{layer_index}/{name}": value
                for layer_index, summary in zip(summary_layer_indices, summaries)
                for name, value in summary.items()
            }

        outputs = tuple(enhanced)
        if return_diagnostics:
            return outputs, tuple(summaries)
        return outputs


def prior_injection_enabled(config: dict) -> bool:
    return bool((config.get("prior_injection", {}) or {}).get("enabled", False))


def build_temporal_prior_injection(
    config: dict,
    feature_channels: int,
    num_layers: int,
) -> TemporalFeaturePyramidPriorInjection | None:
    prior_cfg = config.get("prior_injection", {}) or {}
    if not bool(prior_cfg.get("enabled", False)):
        return None
    method = str(prior_cfg.get("method", "ca_hpi")).lower()
    ca_hpi_methods = {"ca_hpi", "cahpi", "content_aware"}
    spatial_film_methods = {
        "source_aware_spatial_film",
        "source_aware_film",
        "spatial_film",
        "sa_sfilm",
    }
    if method not in ca_hpi_methods | spatial_film_methods:
        raise ValueError(f"Unsupported prior injection method: {method}")
    fusion_mode = (
        "attention_residual"
        if method in ca_hpi_methods
        else "source_aware_spatial_film"
    )

    fusion_cfg = prior_cfg.get("fusion", {}) or {}
    diagnostics_cfg = prior_cfg.get("diagnostics", {}) or {}
    if not isinstance(diagnostics_cfg, dict):
        raise ValueError("prior_injection.diagnostics must be a mapping.")
    token_dim = int(prior_cfg.get("token_dim", 128))
    return TemporalFeaturePyramidPriorInjection(
        vision_dim=int(feature_channels),
        prior_dim=token_dim,
        num_layers=int(num_layers),
        attention_dim=int(fusion_cfg.get("attention_dim", 128)),
        num_heads=int(fusion_cfg.get("num_heads", 4)),
        gate_hidden_dim=int(fusion_cfg.get("gate_hidden_dim", 128)),
        fusion_mode=fusion_mode,
        source_gate_hidden_dim=int(
            fusion_cfg.get(
                "source_gate_hidden_dim", fusion_cfg.get("gate_hidden_dim", 128)
            )
        ),
        film_hidden_dim=int(
            fusion_cfg.get("film_hidden_dim", fusion_cfg.get("gate_hidden_dim", 128))
        ),
        dropout=float(fusion_cfg.get("dropout", 0.0)),
        confidence_bias_scale=float(fusion_cfg.get("confidence_bias_scale", 1.0)),
        source_balance_bias_scale=float(
            fusion_cfg.get("source_balance_bias_scale", 0.0)
        ),
        initial_gate_bias=float(
            fusion_cfg.get(
                "initial_gate_bias",
                -2.0 if fusion_mode == "attention_residual" else -1.0,
            )
        ),
        initial_strength=float(fusion_cfg.get("initial_strength", 0.0)),
        learnable_strength=bool(fusion_cfg.get("learnable_strength", True)),
        record_diagnostics=bool(diagnostics_cfg.get("enabled", False)),
    )
