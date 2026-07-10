from __future__ import annotations

import os
from types import MethodType
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


class GalileoFeatureGrid(NamedTuple):
    features: torch.Tensor
    grid_size: tuple[int, int]
    hidden_state: torch.Tensor
    features_by_layer: tuple[torch.Tensor, ...]


class GalileoHFEncoder(nn.Module):
    """Hugging Face Galileo wrapper for PASTIS batches."""

    def __init__(
        self,
        checkpoint: str | Path,
        patch_size: int = 8,
        freeze: bool = True,
        normalize: bool = True,
        local_files_only: bool = True,
        output_hidden_states: bool = False,
        spatial_token_strategy: str = "auto",
        hidden_layers: list[int] | tuple[int, ...] | None = None,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        self.checkpoint = Path(checkpoint)
        self.patch_size = int(patch_size)
        self.freeze = bool(freeze)
        self.normalize = bool(normalize)
        self.local_files_only = bool(local_files_only)
        self.output_hidden_states = bool(output_hidden_states)
        self.spatial_token_strategy = spatial_token_strategy
        self.hidden_layers = tuple(int(layer) for layer in (hidden_layers or ()))

        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"Missing Galileo checkpoint directory: {self.checkpoint}. "
                "Place the local HF checkpoint there or override encoder.checkpoint."
            )

        cache_dir = Path(".hf_cache").resolve()
        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("HF_MODULES_CACHE", str(cache_dir / "modules"))

        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError("transformers is required for GalileoHFEncoder.") from exc

        self.processor = AutoProcessor.from_pretrained(
            str(self.checkpoint),
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            str(self.checkpoint),
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        self._materialize_flexi_patch_pinvs()
        self._patch_all_true_attention_masks()
        self._validate_hidden_layers()

        if self.freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.model.eval()

        self.hidden_size = hidden_size or self._infer_hidden_size()

    def _validate_hidden_layers(self) -> None:
        if not self.hidden_layers:
            return
        blocks = self._encoder_blocks()
        if blocks is None:
            raise ValueError("Cannot collect Galileo hidden layers: model.encoder.blocks was not found.")
        depth = len(blocks)
        invalid = [layer for layer in self.hidden_layers if layer < 1 or layer > depth]
        if invalid:
            raise ValueError(
                f"Requested Galileo hidden_layers={invalid}, but available layers are 1..{depth}."
            )

    def _materialize_flexi_patch_pinvs(self) -> None:
        """Rebuild deterministic FlexiViT resize matrices left on meta by HF loading."""

        for module in self.model.modules():
            pinvs = getattr(module, "pinvs", None)
            calculate_pinv = getattr(module, "_calculate_pinv", None)
            base_patch_size = getattr(module, "patch_size", None)
            if not isinstance(pinvs, dict) or not callable(calculate_pinv) or base_patch_size is None:
                continue
            for patch_size, pinv in list(pinvs.items()):
                if torch.is_tensor(pinv) and pinv.is_meta:
                    pinvs[patch_size] = calculate_pinv(base_patch_size, patch_size)

    def _encoder_blocks(self):
        encoder = getattr(self.model, "encoder", None)
        return getattr(encoder, "blocks", None)

    def _patch_all_true_attention_masks(self) -> None:
        """Avoid materializing [B, heads, N, N] masks when every token is valid."""

        def patched_forward(attn_module, x, y=None, attn_mask=None):
            batch_size, num_tokens, channels = x.shape
            q = attn_module.q(x)

            if y is None:
                if attn_module.cross_attn:
                    raise AssertionError("Expected self-attention when y is None.")
                k = attn_module.k(x)
                v = attn_module.v(x)
            else:
                if not attn_module.cross_attn:
                    raise AssertionError("Expected cross-attention when y is provided.")
                k = attn_module.k(y)
                v = attn_module.v(y)

            q = q.reshape(batch_size, num_tokens, attn_module.num_heads, -1).transpose(1, 2)
            k = k.reshape(batch_size, k.shape[1], attn_module.num_heads, -1).transpose(1, 2)
            v = v.reshape(batch_size, v.shape[1], attn_module.num_heads, -1).transpose(1, 2)
            q, k = attn_module.q_norm(q), attn_module.k_norm(k)

            if attn_module.fast_attn:
                if attn_mask is not None:
                    if torch.is_tensor(attn_mask) and bool(attn_mask.all()):
                        attn_mask = None
                    else:
                        attn_mask = attn_mask[:, None, None].repeat(
                            (1, attn_module.num_heads, q.shape[-2], 1)
                        )
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=attn_module.attn_drop.p,
                )
            else:
                if attn_mask is not None and not bool(attn_mask.all()):
                    raise NotImplementedError
                q = q * attn_module.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = attn_module.attn_drop(attn)
                x = attn @ v

            x = x.transpose(1, 2).reshape(batch_size, num_tokens, channels)
            x = attn_module.proj(x)
            return attn_module.proj_drop(x)

        for module in self.model.modules():
            if module.__class__.__name__ == "Attention" and hasattr(module, "fast_attn"):
                module.forward = MethodType(patched_forward, module)

    def train(self, mode: bool = True) -> "GalileoHFEncoder":
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def _infer_hidden_size(self) -> int | None:
        config = getattr(self.model, "config", None)
        for name in ("hidden_size", "embed_dim", "dim", "d_model", "encoder_dim"):
            value = getattr(config, name, None)
            if isinstance(value, int):
                return value
        return None

    def forward(self, samples: list[dict]) -> GalileoFeatureGrid:
        if samples and self._can_forward_as_batch(samples):
            return self._forward_many(samples)
        return self._forward_sequential(samples)

    def _forward_sequential(self, samples: list[dict]) -> GalileoFeatureGrid:
        feature_grids = []
        hidden_states = []
        layer_feature_grids: list[list[torch.Tensor]] = [[] for _ in self.hidden_layers]
        grid_size: tuple[int, int] | None = None

        for sample in samples:
            result = self._forward_one(sample)
            feature_grids.append(result.features)
            hidden_states.append(result.hidden_state)
            for layer_index, layer_features in enumerate(result.features_by_layer):
                layer_feature_grids[layer_index].append(layer_features)
            if grid_size is None:
                grid_size = result.grid_size
            elif grid_size != result.grid_size:
                raise ValueError(f"Mixed Galileo grid sizes in one batch: {grid_size} and {result.grid_size}")

        features = torch.cat(feature_grids, dim=0)
        if all(hidden.shape == hidden_states[0].shape for hidden in hidden_states):
            hidden_state = torch.cat(hidden_states, dim=0)
        else:
            hidden_state = features.new_empty(0)
        features_by_layer = tuple(torch.cat(layer_features, dim=0) for layer_features in layer_feature_grids)

        return GalileoFeatureGrid(
            features=features,
            grid_size=grid_size or (0, 0),
            hidden_state=hidden_state,
            features_by_layer=features_by_layer,
        )

    @staticmethod
    def _can_forward_as_batch(samples: list[dict]) -> bool:
        if not samples:
            return False
        first_shape = tuple(samples[0]["s2"].shape)
        first_months_shape = tuple(samples[0]["months"].shape)
        return all(
            tuple(sample["s2"].shape) == first_shape
            and tuple(sample["months"].shape) == first_months_shape
            for sample in samples
        )

    def _forward_many(self, samples: list[dict]) -> GalileoFeatureGrid:
        s2 = samples[0]["s2"]
        if s2.ndim != 4:
            raise ValueError(f"Expected S2 [T, C, H, W], got {tuple(s2.shape)}")

        timesteps, _, height, width = s2.shape
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Image size {(height, width)} is not divisible by patch_size={self.patch_size}"
            )

        model_device = next(self.model.parameters()).device
        processor_inputs = self._build_batched_processor_inputs(samples)
        processor_inputs = self._move_to_device(processor_inputs, model_device)

        captured_hidden_layers: dict[int, torch.Tensor] = {}
        hooks = self._register_hidden_layer_hooks(captured_hidden_layers)
        try:
            with torch.set_grad_enabled(not self.freeze):
                try:
                    outputs = self.model(
                        **processor_inputs,
                        output_hidden_states=self.output_hidden_states,
                        return_dict=True,
                    )
                except TypeError:
                    outputs = self.model(**processor_inputs)
        finally:
            for hook in hooks:
                hook.remove()

        hidden = self._extract_hidden(outputs)
        spatial_tokens = self._select_spatial_tokens(
            hidden,
            grid_h=grid_h,
            grid_w=grid_w,
            timesteps=int(timesteps),
        )
        features = spatial_tokens.transpose(1, 2).reshape(
            len(samples),
            spatial_tokens.shape[-1],
            grid_h,
            grid_w,
        )
        features_by_layer = tuple(
            self._hidden_to_feature_grid(
                captured_hidden_layers[layer],
                grid_h=grid_h,
                grid_w=grid_w,
                timesteps=int(timesteps),
            )
            for layer in self.hidden_layers
        )
        return GalileoFeatureGrid(
            features=features,
            grid_size=(grid_h, grid_w),
            hidden_state=hidden,
            features_by_layer=features_by_layer,
        )

    def _forward_one(self, sample: dict) -> GalileoFeatureGrid:
        s2 = sample["s2"]
        months = sample["months"]
        if s2.ndim != 4:
            raise ValueError(f"Expected S2 [T, C, H, W], got {tuple(s2.shape)}")

        _, _, height, width = s2.shape
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Image size {(height, width)} is not divisible by patch_size={self.patch_size}"
            )

        model_device = next(self.model.parameters()).device
        processor_inputs = self._build_processor_inputs(s2=s2, months=months)
        processor_inputs = self._move_to_device(processor_inputs, model_device)

        captured_hidden_layers: dict[int, torch.Tensor] = {}
        hooks = self._register_hidden_layer_hooks(captured_hidden_layers)
        try:
            with torch.set_grad_enabled(not self.freeze):
                try:
                    outputs = self.model(
                        **processor_inputs,
                        output_hidden_states=self.output_hidden_states,
                        return_dict=True,
                    )
                except TypeError:
                    outputs = self.model(**processor_inputs)
        finally:
            for hook in hooks:
                hook.remove()

        hidden = self._extract_hidden(outputs)
        spatial_tokens = self._select_spatial_tokens(
            hidden,
            grid_h=grid_h,
            grid_w=grid_w,
            timesteps=int(s2.shape[0]),
        )
        features = spatial_tokens.transpose(1, 2).reshape(1, spatial_tokens.shape[-1], grid_h, grid_w)
        features_by_layer = tuple(
            self._hidden_to_feature_grid(
                captured_hidden_layers[layer],
                grid_h=grid_h,
                grid_w=grid_w,
                timesteps=int(s2.shape[0]),
            )
            for layer in self.hidden_layers
        )
        return GalileoFeatureGrid(
            features=features,
            grid_size=(grid_h, grid_w),
            hidden_state=hidden,
            features_by_layer=features_by_layer,
        )

    def _register_hidden_layer_hooks(self, captured: dict[int, torch.Tensor]):
        if not self.hidden_layers:
            return []
        blocks = self._encoder_blocks()
        if blocks is None:
            raise ValueError("Cannot collect Galileo hidden layers: model.encoder.blocks was not found.")

        hooks = []
        for layer in self.hidden_layers:
            block = blocks[layer - 1]

            def capture_hidden(_module, _inputs, output, layer=layer):
                if isinstance(output, (tuple, list)):
                    output = output[0]
                if not torch.is_tensor(output):
                    raise TypeError(f"Galileo layer {layer} returned non-tensor output: {type(output)}")
                captured[layer] = output

            hooks.append(block.register_forward_hook(capture_hidden))
        return hooks

    def _hidden_to_feature_grid(
        self,
        hidden: torch.Tensor,
        grid_h: int,
        grid_w: int,
        timesteps: int,
    ) -> torch.Tensor:
        spatial_tokens = self._select_spatial_tokens(
            hidden,
            grid_h=grid_h,
            grid_w=grid_w,
            timesteps=timesteps,
        )
        return spatial_tokens.transpose(1, 2).reshape(
            hidden.shape[0],
            spatial_tokens.shape[-1],
            grid_h,
            grid_w,
        )

    def _build_processor_inputs(self, s2: torch.Tensor, months: torch.Tensor):
        s2_hw_t_c = s2.permute(2, 3, 0, 1).contiguous().cpu().numpy()
        kwargs = {
            "s2": s2_hw_t_c,
            "months": months.cpu().numpy(),
            "normalize": self.normalize,
            "patch_size": self.patch_size,
        }
        try:
            inputs = self.processor(**kwargs)
        except TypeError:
            kwargs.pop("patch_size")
            inputs = self.processor(**kwargs)
        return inputs

    def _build_batched_processor_inputs(self, samples: list[dict]):
        processed = [
            self._build_processor_inputs(s2=sample["s2"], months=sample["months"])
            for sample in samples
        ]
        first = dict(processed[0])
        batched = {}
        for key in first:
            values = [dict(inputs)[key] for inputs in processed]
            first_value = values[0]
            if key == "patch_size":
                batched[key] = first_value
                continue
            if torch.is_tensor(first_value):
                try:
                    batched[key] = torch.cat(values, dim=0)
                except RuntimeError as exc:
                    shapes = [tuple(value.shape) for value in values]
                    raise ValueError(f"Cannot batch Galileo processor input {key}: {shapes}") from exc
            else:
                if any(value != first_value for value in values):
                    raise ValueError(f"Cannot batch differing Galileo processor input {key}: {values}")
                batched[key] = first_value
        return batched

    @staticmethod
    def _move_to_device(inputs, device: torch.device):
        if hasattr(inputs, "to"):
            return inputs.to(device)
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in dict(inputs).items()
        }

    @staticmethod
    def _extract_hidden(outputs) -> torch.Tensor:
        for name in ("last_hidden_state", "encoder_last_hidden_state", "hidden_state"):
            value = getattr(outputs, name, None)
            if torch.is_tensor(value):
                return value
        if isinstance(outputs, dict):
            for name in ("last_hidden_state", "encoder_last_hidden_state", "hidden_state"):
                value = outputs.get(name)
                if torch.is_tensor(value):
                    return value
        if isinstance(outputs, (tuple, list)) and outputs and torch.is_tensor(outputs[0]):
            return outputs[0]
        raise RuntimeError("Could not find a token tensor in Galileo model outputs.")

    def _select_spatial_tokens(
        self,
        hidden: torch.Tensor,
        grid_h: int,
        grid_w: int,
        timesteps: int,
    ) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"Expected hidden state [B, N, D], got {tuple(hidden.shape)}")
        grid_tokens = grid_h * grid_w
        token_count = hidden.shape[1]
        if token_count < grid_tokens:
            raise ValueError(
                f"Galileo returned {token_count} tokens, fewer than spatial grid tokens {grid_tokens}"
            )

        strategy = self.spatial_token_strategy
        if strategy == "all":
            if token_count != grid_tokens:
                raise ValueError(f"Expected exactly {grid_tokens} tokens, got {token_count}")
            return hidden
        if strategy == "drop_cls":
            return hidden[:, 1 : 1 + grid_tokens]
        if strategy == "first":
            return hidden[:, :grid_tokens]
        if strategy == "last":
            return hidden[:, -grid_tokens:]
        if strategy == "spacetime_mean":
            return self._aggregate_spacetime_tokens(
                hidden,
                grid_h=grid_h,
                grid_w=grid_w,
                timesteps=timesteps,
            )
        if strategy != "auto":
            raise ValueError(f"Unknown spatial_token_strategy: {strategy}")

        if token_count == grid_tokens:
            return hidden
        if token_count == grid_tokens + 1:
            return hidden[:, 1:]
        return self._aggregate_spacetime_tokens(
            hidden,
            grid_h=grid_h,
            grid_w=grid_w,
            timesteps=timesteps,
        )

    @staticmethod
    def _aggregate_spacetime_tokens(
        hidden: torch.Tensor,
        grid_h: int,
        grid_w: int,
        timesteps: int,
    ) -> torch.Tensor:
        """Aggregate Galileo space-time tokens into one token per spatial patch.

        Galileo's sequence can contain one token per spatial patch, timestep, and
        modality/band group. For PASTIS S2 this is expected to be laid out as
        [H_grid, W_grid, T, group] before flattening. We therefore reduce over
        T and group, not over arbitrary contiguous chunks of length H*W.

        If the returned sequence does not match this structure, fail loudly
        instead of silently reshaping mixed token types into a wrong feature map.
        """
        if timesteps <= 0:
            raise ValueError(f"timesteps must be positive, got {timesteps}")

        grid_tokens = grid_h * grid_w
        token_count = hidden.shape[1]
        tokens_per_group = grid_tokens * timesteps
        if token_count % tokens_per_group != 0:
            raise ValueError(
                "Cannot safely recover a spatial feature grid from Galileo tokens: "
                f"token_count={token_count}, grid_tokens={grid_tokens}, timesteps={timesteps}. "
                "Set encoder.spatial_token_strategy to an explicit strategy only after "
                "verifying the Galileo token layout."
            )

        group_count = token_count // tokens_per_group
        grouped = hidden.reshape(
            hidden.shape[0],
            grid_h,
            grid_w,
            timesteps,
            group_count,
            hidden.shape[-1],
        )
        spatial = grouped.mean(dim=(3, 4))
        return spatial.reshape(hidden.shape[0], grid_tokens, hidden.shape[-1])
