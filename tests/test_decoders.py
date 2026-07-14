from __future__ import annotations

import unittest

import torch

from models import build_cached_feature_model
from models.decoders import (
    DPTMultiLayerDecoder,
    DPTSingleLayerDecoder,
    GalileoDPTDecoder,
    GalileoLinearProbeDecoder,
    UPerNetDecoder,
)


class BaselineResolutionTest(unittest.TestCase):
    def test_single_and_multi_layer_baselines_keep_the_native_grid(self) -> None:
        single = DPTSingleLayerDecoder(
            in_channels=8,
            num_classes=5,
            decoder_channels=16,
            decoder_blocks=1,
        )
        multi = DPTMultiLayerDecoder(
            in_channels=8,
            num_classes=5,
            num_layers=4,
            decoder_channels=16,
            decoder_blocks=1,
            fusion_blocks=1,
        )
        native_shapes = []
        hooks = [
            single.refine.register_forward_hook(
                lambda _module, _inputs, output: native_shapes.append(tuple(output.shape[-2:]))
            ),
            multi.refine.register_forward_hook(
                lambda _module, _inputs, output: native_shapes.append(tuple(output.shape[-2:]))
            ),
        ]

        single_logits = single(torch.randn(1, 8, 8, 8), target_size=(32, 32))
        multi_logits = multi(
            tuple(torch.randn(1, 8, 8, 8) for _ in range(4)),
            target_size=(32, 32),
        )
        for hook in hooks:
            hook.remove()

        self.assertEqual(native_shapes, [(8, 8), (8, 8)])
        self.assertEqual(tuple(single_logits.shape), (1, 5, 32, 32))
        self.assertEqual(tuple(multi_logits.shape), (1, 5, 32, 32))


class GalileoDPTDecoderTest(unittest.TestCase):
    def test_reassembles_four_galileo_layers_and_backpropagates(self) -> None:
        config = {
            "data": {"num_classes": 5},
            "encoder": {"hidden_layers": [3, 6, 9, 12]},
            "model": {
                "decoder": "galileo_dpt",
                "decoder_channels": 16,
                "fusion_blocks": 1,
                "head_channels": 8,
                "dropout": 0.0,
            },
        }
        model = build_cached_feature_model(config, in_channels=8, num_layers=4)
        batch = {
            "features": torch.randn(2, 8, 8, 8),
            "features_by_layer": torch.randn(2, 4, 8, 8, 8),
            "target": torch.zeros(2, 32, 32, dtype=torch.long),
        }
        pyramid_shapes = []
        deepest_refine_shapes = []
        hooks = [
            adapter.register_forward_hook(
                lambda _module, _inputs, output: pyramid_shapes.append(tuple(output.shape))
            )
            for adapter in model.decoder.reassemble
        ]
        hooks.append(
            model.decoder.reassemble[-1].refine.register_forward_hook(
                lambda _module, _inputs, output: deepest_refine_shapes.append(
                    tuple(output.shape)
                )
            )
        )

        logits = model(batch)
        for hook in hooks:
            hook.remove()

        self.assertIsInstance(model.decoder, GalileoDPTDecoder)
        self.assertEqual(
            pyramid_shapes,
            [
                (2, 16, 32, 32),
                (2, 16, 16, 16),
                (2, 16, 8, 8),
                (2, 16, 4, 4),
            ],
        )
        self.assertEqual(
            deepest_refine_shapes,
            [(2, 16, 4, 4), (2, 16, 8, 8)],
        )
        self.assertEqual(tuple(logits.shape), (2, 5, 32, 32))
        self.assertTrue(torch.isfinite(logits).all())
        logits.mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_native_skip_keeps_legacy_state_dict_keys(self) -> None:
        legacy = GalileoDPTDecoder(
            in_channels=8,
            num_classes=5,
            decoder_channels=16,
            fusion_blocks=1,
            head_channels=8,
            preserve_native_deep_skip=False,
        )
        current = GalileoDPTDecoder(
            in_channels=8,
            num_classes=5,
            decoder_channels=16,
            fusion_blocks=1,
            head_channels=8,
            preserve_native_deep_skip=True,
        )

        self.assertEqual(set(legacy.state_dict()), set(current.state_dict()))
        current.load_state_dict(legacy.state_dict(), strict=True)

    def test_requires_exactly_four_ordered_layers(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires exactly four"):
            GalileoDPTDecoder(
                in_channels=8,
                num_classes=5,
                num_layers=3,
                decoder_channels=16,
                fusion_blocks=1,
                head_channels=8,
            )

    def test_requires_at_least_one_fusion_block(self) -> None:
        with self.assertRaisesRegex(ValueError, "fusion_blocks must be at least one"):
            GalileoDPTDecoder(
                in_channels=8,
                num_classes=5,
                num_layers=4,
                decoder_channels=16,
                fusion_blocks=0,
                head_channels=8,
            )


class UPerNetDecoderTest(unittest.TestCase):
    def test_cached_model_forward_and_backward(self) -> None:
        config = {
            "data": {"num_classes": 5},
            "encoder": {"hidden_layers": [3, 6, 9, 12]},
            "model": {
                "decoder": "upernet",
                "decoder_channels": 16,
                "ppm_channels": 4,
                "ppm_scales": [1, 2, 3, 6],
                "dropout": 0.0,
            },
        }
        model = build_cached_feature_model(config, in_channels=8, num_layers=4)
        batch = {
            "features": torch.randn(1, 8, 8, 8),
            "features_by_layer": torch.randn(1, 4, 8, 8, 8),
            "target": torch.zeros(1, 32, 32, dtype=torch.long),
        }

        ppm_inputs = []
        hook = model.decoder.ppm.bottleneck.register_forward_pre_hook(
            lambda _module, inputs: ppm_inputs.append(inputs[0].detach().clone())
        )
        logits = model(batch)
        hook.remove()
        self.assertEqual(tuple(logits.shape), (1, 5, 32, 32))
        self.assertTrue(
            torch.equal(ppm_inputs[0][:, :8], batch["features_by_layer"][:, -1])
        )
        self.assertTrue(torch.isfinite(logits).all())
        logits.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_requires_multiple_layers(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            UPerNetDecoder(
                in_channels=8,
                num_classes=5,
                num_layers=1,
                decoder_channels=16,
                ppm_channels=4,
            )


class GalileoLinearProbeTest(unittest.TestCase):
    def test_patch_tokens_map_directly_to_pixel_logits(self) -> None:
        config = {
            "data": {"num_classes": 5},
            "encoder": {"hidden_layers": [3, 6, 9, 12]},
            "model": {"decoder": "linear_probe", "output_patch_size": 4},
        }
        model = build_cached_feature_model(config, in_channels=8)
        batch = {
            "features": torch.randn(2, 8, 8, 8),
            "target": torch.zeros(2, 32, 32, dtype=torch.long),
        }

        logits = model(batch)

        self.assertEqual(tuple(logits.shape), (2, 5, 32, 32))
        self.assertIsInstance(model.decoder, GalileoLinearProbeDecoder)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            8 * (5 * 4 * 4) + (5 * 4 * 4),
        )
        logits.mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
