import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from atlas import ATLAS, ATLASSegmentationAdapter

try:
    from atlas.vlm_instance import ATLASInstance
except (ImportError, OSError):
    ATLASInstance = None
    VLM_DEPENDENCY_AVAILABLE = False
else:
    VLM_DEPENDENCY_AVAILABLE = True
from atlas.common import (
    DIGBuffer,
    active_unit_count,
    asymmetric_clip_ratio,
    compute_sos_penalty,
    quantile_selection_mask,
    softmax_entropy,
)


class TinyViT(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=8):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.num_patches = 4
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(hidden_dim),
                        "norm2": nn.LayerNorm(hidden_dim),
                    }
                )
                for _ in range(12)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.head(self.norm(x))


class TinyGNNet(nn.Module):
    def __init__(self, num_classes=3, channels=4):
        super().__init__()
        self.layer1 = nn.Sequential(nn.GroupNorm(1, channels))
        self.head = nn.Linear(channels * 4 * 4, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        return self.head(x.view(x.shape[0], -1))


class TinyPromptLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx = nn.Parameter(torch.tensor([[0.2, -0.1]], dtype=torch.float32))

    def reset(self):
        with torch.no_grad():
            self.ctx.copy_(self.ctx_init_state)


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_learner = TinyPromptLearner()
        self.classnames = ["alpha", "beta"]

    def forward(self, images):
        feature = images.view(images.shape[0], -1).mean(dim=1, keepdim=True)
        prompt = self.prompt_learner.ctx.mean()
        return torch.cat([feature + prompt, -feature - prompt], dim=1)

    def reset(self):
        self.prompt_learner.reset()


class TinySegformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"num_labels": 3, "model_type": "segformer"})()
        self.norm = nn.LayerNorm(4)

    def forward(self, pixel_values):
        features = pixel_values.permute(0, 2, 3, 1)
        features = self.norm(features).permute(0, 3, 1, 2)
        return type("Output", (), {"logits": features[:, :3]})()


class TinyArgs:
    lr = 0.05
    adadem_pi = 0.1
    atlas_adadem_weight = 0.0
    atlas_entropy_weight = 0.35
    atlas_source_anchor_weight = 0.02
    atlas_view_consistency_weight = 0.05
    atlas_source_output = "prompt"
    atlas_output_fusion = "confidence"
    atlas_source_output_weight = 1.0
    atlas_adapted_output_weight = 1.0
    atlas_view_output_weight = 0.5
    atlas_fusion_temperature = 2.0
    atlas_source_ensemble_anchor = False
    prompt_ensemble_chunk_size = 256
    selection_p = 0.5
    tta_steps = 1
    atlas_method_variant = "full"


class TestATLASCore(unittest.TestCase):
    def test_quantile_selection_uses_only_unlabeled_score_order(self):
        scores = torch.tensor([0.1, 0.4, 0.2, 0.9])
        self.assertEqual(
            quantile_selection_mask(scores, 0.5).tolist(),
            [False, True, False, True],
        )

    def test_quantile_selection_rejects_invalid_quantile(self):
        with self.assertRaises(ValueError):
            quantile_selection_mask(torch.tensor([0.1]), 1.1)


    def test_method_variant_defaults_to_full_and_rejects_unknown_values(self):
        full = self._make_gn_adapter()
        self.assertEqual(full.method_variant, "full")
        self.assertTrue(full.use_branch_refinements)

        with self.assertRaisesRegex(ValueError, "method_variant"):
            ATLAS(
                TinyGNNet(),
                adapt_type="gn",
                method_variant="experimental",
                num_classes=3,
            )

    def test_classification_core_disables_branch_refinements_and_bs1_buffer(self):
        adapter = ATLAS(
            TinyViT(num_classes=3, hidden_dim=8),
            adapt_type="ln",
            adaptation_mode="continual",
            scenario="bs1",
            method_variant="core",
            num_classes=3,
        )

        self.assertEqual(adapter.method_variant, "core")
        self.assertFalse(adapter.use_branch_refinements)
        self.assertEqual(adapter.uan_mode, "sample")
        self.assertFalse(adapter.buffering_enabled)
        self.assertFalse(adapter.ln_bs1_loss_enabled)
        self.assertFalse(adapter.ln_natural_shift_enabled)

    def test_classification_core_uses_ln_base_path_with_shared_roles(self):
        core = ATLAS(
            TinyViT(num_classes=3, hidden_dim=8),
            adapt_type="ln",
            adaptation_mode="continual",
            method_variant="core",
            num_classes=3,
        )
        x = torch.randn(2, 1, 2, 4)
        self.assertTrue(core.ln_shared_role_only)
        with patch.object(core, "_adapt_on_batch_ln_vit", return_value=True) as ln_path:
            self.assertTrue(core._adapt_on_batch(x))
        ln_path.assert_called_once_with(x)

        full = self._make_ln_adapter()
        self.assertFalse(full.ln_shared_role_only)
        with patch.object(full, "_adapt_on_batch_ln_vit", return_value=True) as ln_path:
            self.assertTrue(full._adapt_on_batch(x))
        ln_path.assert_called_once_with(x)

    def test_segmentation_core_disables_optional_output_engineering(self):
        adapter = ATLASSegmentationAdapter(
            TinySegformer(),
            method_variant="core",
            vit_ln_config={
                "output_hflip": True,
                "output_scale_weight": 1.0,
                "adapt_decode_bn": True,
            },
        )

        self.assertEqual(adapter.method_variant, "core")
        self.assertFalse(adapter.ln_output_hflip)
        self.assertEqual(adapter.ln_output_scale_weight, 0.0)
        self.assertFalse(adapter.ln_adapt_decode_bn)

    @unittest.skipUnless(VLM_DEPENDENCY_AVAILABLE, "VLM-TTA/DEM baseline is an external dependency")
    def test_vlm_core_keeps_anchor_but_disables_auxiliary_losses_and_fusion(self):
        args = TinyArgs()
        args.atlas_method_variant = "core"
        trainer = ATLASInstance(TinyVLM(), torch.device("cpu"))
        trainer.prepare_model_and_optimization(args)

        self.assertEqual(trainer.method_variant, "core")
        self.assertGreater(trainer.anchor_weight, 0.0)
        self.assertEqual(trainer.entropy_weight, 0.0)
        self.assertEqual(trainer.view_consistency_weight, 0.0)
        self.assertEqual(trainer.output_fusion, "adapted")
    def _make_ln_adapter(
        self,
        enabled=False,
        adaptation_mode="continual",
        num_classes=3,
        param_scope="full_ln",
        vit_ln_overrides=None,
    ):
        vit_ln_config = {
            "param_scope": param_scope,
            "natural_shift": {"enabled": enabled},
        }
        if vit_ln_overrides:
            for key, value in vit_ln_overrides.items():
                if key == "natural_shift":
                    merged = deepcopy(vit_ln_config["natural_shift"])
                    merged.update(value)
                    vit_ln_config["natural_shift"] = merged
                else:
                    vit_ln_config[key] = value
        return ATLAS(
            TinyViT(num_classes=num_classes, hidden_dim=8),
            adapt_type="ln",
            adaptation_mode=adaptation_mode,
            scenario="normal",
            num_classes=num_classes,
            vit_ln_config=vit_ln_config,
            no_arc=True,
            no_dig=True,
            no_uan=True,
            no_sos=True,
        )

    def _make_gn_adapter(self, active_views=None, reward_config=None):
        kwargs = {}
        if active_views is not None:
            kwargs["view_config"] = {"active": active_views}
        if reward_config is not None:
            kwargs["reward_config"] = reward_config
        return ATLAS(
            TinyGNNet(num_classes=3, channels=4),
            adapt_type="gn",
            adaptation_mode="continual",
            scenario="normal",
            num_classes=3,
            **kwargs,
        )

    def _patch_group_signal_views(
        self,
        adapter,
        x,
        *,
        source_logits,
        weak_logits,
        structural_logits,
        destroy_logits,
    ):
        outputs = [weak_logits, structural_logits, destroy_logits]

        def fake_model_forward(_):
            return outputs.pop(0)

        adapter.model.forward = fake_model_forward
        adapter.source_model.forward = lambda _: source_logits

    def test_arc_clip_bounds(self):
        ratio = torch.tensor([0.50, 0.95, 1.10, 1.60])
        clipped = asymmetric_clip_ratio(ratio.clone(), eps_low=0.2, eps_high=0.28)
        expected = torch.tensor([0.80, 0.95, 1.10, 1.28])
        self.assertTrue(torch.allclose(clipped, expected))

    def test_dig_buffer_accumulates_bs1_until_ready(self):
        buffer = DIGBuffer(target_size=3)
        buffer.add([torch.tensor([1.0, 2.0])])
        buffer.add([torch.tensor([3.0, 4.0])])
        self.assertFalse(buffer.ready())
        self.assertIsNone(buffer.pop_ready_batch(torch.device("cpu")))

        buffer.add([torch.tensor([5.0, 6.0])])
        self.assertTrue(buffer.ready())
        batch = buffer.pop_ready_batch(torch.device("cpu"))
        self.assertEqual(batch.shape, (3, 2))
        self.assertEqual(len(buffer), 0)

    def test_uan_active_unit_count_modes(self):
        self.assertEqual(active_unit_count(batch_size=3, mode="sample"), 3)
        self.assertEqual(active_unit_count(batch_size=2, mode="token", token_count=197), 394)
        self.assertEqual(active_unit_count(batch_size=1, mode="pixel", spatial_shape=(8, 8)), 64)
        self.assertEqual(
            active_unit_count(batch_size=2, mode="prompt_view", token_count=4, prompt_views=3),
            24,
        )

    def test_sos_penalty_piecewise_behavior(self):
        score = torch.tensor([0.10, 0.55, 0.90])
        penalty = compute_sos_penalty(score, safe_margin=0.35, hard_margin=0.75)
        expected = torch.tensor([0.0, 0.5, 1.0])
        self.assertTrue(torch.allclose(penalty, expected))

    @unittest.skipUnless(VLM_DEPENDENCY_AVAILABLE, "VLM-TTA/DEM baseline is an external dependency")
    def test_vlm_instance_resets_prompt_to_source_state_each_sample(self):
        model = TinyVLM()
        trainer = ATLASInstance(model, torch.device("cpu"))
        trainer.prepare_model_and_optimization(TinyArgs())
        source_ctx = model.prompt_learner.ctx.detach().clone()

        with torch.no_grad():
            model.prompt_learner.ctx.add_(9.0)
        trainer.pre_adaptation()

        self.assertTrue(torch.allclose(model.prompt_learner.ctx, source_ctx))

    @unittest.skipUnless(VLM_DEPENDENCY_AVAILABLE, "VLM-TTA/DEM baseline is an external dependency")
    def test_vlm_instance_returns_original_image_logits_after_view_adaptation(self):
        model = TinyVLM()
        trainer = ATLASInstance(model, torch.device("cpu"))
        trainer.prepare_model_and_optimization(TinyArgs())
        trainer.pre_adaptation()

        image = torch.ones(1, 1, 2, 2)
        views = torch.cat([image, image * 0.5, image * -0.5], dim=0)
        result = trainer.adaptation_process(image, views, TinyArgs())

        self.assertEqual(set(result), {"output"})
        self.assertEqual(result["output"].shape, (1, 2))
        self.assertTrue(torch.isfinite(result["output"]).all())

    @unittest.skipUnless(VLM_DEPENDENCY_AVAILABLE, "VLM-TTA/DEM baseline is an external dependency")
    def test_vlm_instance_confidence_fusion_prefers_lower_entropy_candidate(self):
        adapted = torch.tensor([[0.1, 0.0]])
        source = torch.tensor([[4.0, -4.0]])
        view = torch.tensor([[0.0, 0.1]])

        fused = ATLASInstance.fuse_logits_by_confidence(
            adapted_logits=adapted,
            source_logits=source,
            view_logits=view,
            adapted_weight=1.0,
            source_weight=1.0,
            view_weight=1.0,
            temperature=2.0,
        )

        self.assertEqual(int(fused.argmax(dim=1).item()), 0)
        self.assertGreater(fused.softmax(dim=1)[0, 0].item(), 0.5)

    @unittest.skipUnless(VLM_DEPENDENCY_AVAILABLE, "VLM-TTA/DEM baseline is an external dependency")
    def test_vlm_instance_uses_ensemble_source_for_zero_shot_anchor(self):
        model = TinyVLM()
        trainer = ATLASInstance(model, torch.device("cpu"))
        trainer.prepare_model_and_optimization(TinyArgs())
        trainer.source_output_mode = "ensemble"
        trainer.source_ensemble_text_features = object()
        images = torch.ones(2, 1, 2, 2)
        ensemble_logits = torch.tensor([[3.0, -1.0], [2.0, -2.0]])

        with patch("atlas.vlm_instance.source_ensemble_logits", return_value=ensemble_logits) as mocked:
            source = trainer._forward_anchor_source_logits(images)

        mocked.assert_called_once()
        self.assertTrue(torch.equal(source, ensemble_logits))

    def test_atlas_online_is_not_exported_from_live_api(self):
        import atlas

        self.assertNotIn("ATLASOnline", atlas.__all__)
        with self.assertRaises(AttributeError):
            getattr(atlas, "ATLASOnline")

    def test_dig_skips_zero_variance_groups(self):
        model = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 2))
        adapter = ATLAS(
            model,
            adapt_type="bn",
            scenario="normal",
            num_classes=2,
            dig_config={"std_threshold": 0.05},
        )

        def fake_group_signals(x, raw_logits):
            batch = x.shape[0]
            source_logits = torch.zeros_like(raw_logits)
            raw_probs = torch.full_like(raw_logits, 0.5)
            source_probs = torch.full_like(raw_logits, 0.5)
            pseudo_targets = torch.zeros(batch, dtype=torch.long, device=x.device)
            raw_advantage = torch.ones(batch, device=x.device)
            rewards = torch.zeros(batch, 4, device=x.device)
            reward_std = torch.zeros(batch, device=x.device)
            sos_weight = torch.ones(batch, device=x.device)
            aux_info = {
                "raw_probs": raw_probs,
                "source_probs": source_probs,
                "raw_pred": torch.zeros(batch, dtype=torch.long, device=x.device),
                "source_pred": torch.zeros(batch, dtype=torch.long, device=x.device),
                "destroy_sensitivity": torch.zeros(batch, device=x.device),
                "raw_entropy": torch.zeros(batch, device=x.device),
                "entropy_score": torch.ones(batch, device=x.device),
            }
            return reward_std, sos_weight, pseudo_targets, source_logits, raw_advantage, rewards, aux_info

        adapter._compute_group_signals = fake_group_signals
        x = torch.randn(2, 4)
        self.assertFalse(adapter._adapt_on_batch(x))
        self.assertEqual(adapter.update_count, 0)

    def test_bs1_buffered_adaptation_waits_for_target_size(self):
        model = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 2))
        adapter = ATLAS(
            model,
            adapt_type="bn",
            scenario="bs1",
            num_classes=2,
            dig_config={"buffer_target": 2, "buffer_scenarios": ["bs1"], "std_threshold": 0.01},
        )

        seen_batches = []

        adapter._precompute_informative_mask = lambda x: torch.ones(x.shape[0], dtype=torch.bool)

        def fake_adapt(batch):
            seen_batches.append(batch.shape[0])
            return True

        adapter._adapt_on_batch = fake_adapt

        adapter._maybe_adapt(torch.randn(1, 4))
        self.assertEqual(seen_batches, [])
        self.assertEqual(adapter.dig_buffer_fills, 1)
        self.assertEqual(adapter.skip_count, 1)

        adapter._maybe_adapt(torch.randn(1, 4))
        self.assertEqual(seen_batches, [2])
        self.assertEqual(adapter.dig_buffer_fills, 2)

    def test_ln_natural_shift_bridge_activation_is_continual_only(self):
        self.assertFalse(self._make_ln_adapter(enabled=False)._use_ln_natural_shift_bridge())
        self.assertTrue(self._make_ln_adapter(enabled=True)._use_ln_natural_shift_bridge())
        self.assertFalse(
            self._make_ln_adapter(
                enabled=True,
                adaptation_mode="standard",
            )._use_ln_natural_shift_bridge()
        )

    def test_gn_active_views_require_raw(self):
        with self.assertRaisesRegex(ValueError, "must include 'raw'"):
            self._make_gn_adapter(active_views=["source", "photometric"])

    def test_gn_disabling_source_removes_anchor_penalty_from_sos(self):
        x = torch.randn(2, 4, 4, 4)
        raw_logits = torch.tensor(
            [[5.0, 0.2, -1.0], [4.5, 0.1, -0.5]],
            dtype=torch.float32,
        )
        source_logits = torch.tensor(
            [[0.1, 4.8, -1.2], [0.2, 4.6, -0.7]],
            dtype=torch.float32,
        )
        weak_logits = raw_logits.clone()
        structural_logits = raw_logits.clone()
        destroy_logits = raw_logits.clone()

        adapter_full = self._make_gn_adapter(
            active_views=["source", "raw", "photometric", "structural", "destroy"]
        )
        self._patch_group_signal_views(
            adapter_full,
            x,
            source_logits=source_logits,
            weak_logits=weak_logits,
            structural_logits=structural_logits,
            destroy_logits=destroy_logits,
        )
        _, sos_full, _, _, _, _, _ = adapter_full._compute_group_signals(x, raw_logits)

        adapter_no_source = self._make_gn_adapter(
            active_views=["raw", "photometric", "structural", "destroy"]
        )
        self._patch_group_signal_views(
            adapter_no_source,
            x,
            source_logits=source_logits,
            weak_logits=weak_logits,
            structural_logits=structural_logits,
            destroy_logits=destroy_logits,
        )
        _, sos_no_source, _, _, _, _, _ = adapter_no_source._compute_group_signals(x, raw_logits)

        self.assertTrue(torch.all(sos_no_source >= sos_full))
        self.assertEqual(adapter_no_source.get_stats()["view_selection_mode"], "explicit")
        self.assertEqual(
            adapter_no_source.get_stats()["active_views"],
            ["raw", "photometric", "structural", "destroy"],
        )

    def test_eq7_default_reward_is_equal_mean(self):
        x = torch.randn(2, 4, 4, 4)
        raw_logits = torch.tensor([[3.0, 0.4, -0.2], [0.3, 2.5, -0.5]])
        source_logits = torch.tensor([[2.8, 0.5, -0.3], [2.2, 0.4, -0.2]])
        weak_logits = torch.tensor([[2.7, 0.6, -0.1], [0.2, 2.3, -0.4]])
        structural_logits = torch.tensor([[0.5, 2.4, -0.3], [0.4, 2.1, -0.2]])
        destroy_logits = torch.tensor([[2.2, 0.8, -0.2], [1.9, 0.6, -0.1]])
        adapter = self._make_gn_adapter()
        self._patch_group_signal_views(
            adapter,
            x,
            source_logits=source_logits,
            weak_logits=weak_logits,
            structural_logits=structural_logits,
            destroy_logits=destroy_logits,
        )

        _, _, _, _, _, rewards, aux = adapter._compute_group_signals(x, raw_logits)
        expected = torch.stack(
            (
                aux["target_prob_rewards"],
                aux["entropy_rewards"],
                aux["source_agreement_rewards"],
                aux["consensus_rewards"],
            ),
            dim=-1,
        ).mean(dim=-1)

        self.assertTrue(torch.equal(rewards, expected))
        self.assertEqual(
            adapter.get_stats()["reward_aggregation"],
            "equal_mean",
        )
        self.assertIsNone(adapter.get_stats()["legacy_reward_weights"])

    def test_eq7_custom_reward_weights_control_reward_tensor(self):
        weights = {"target": 0.54, "entropy": 0.167273, "source": 0.167273, "consensus": 0.125454}
        x = torch.randn(2, 4, 4, 4)
        raw_logits = torch.tensor([[3.0, 0.4, -0.2], [0.3, 2.5, -0.5]])
        adapter = self._make_gn_adapter(
            reward_config={"aggregation": "legacy_weighted", "legacy_weights": weights}
        )
        self._patch_group_signal_views(
            adapter,
            x,
            source_logits=torch.tensor([[2.8, 0.5, -0.3], [2.2, 0.4, -0.2]]),
            weak_logits=torch.tensor([[2.7, 0.6, -0.1], [0.2, 2.3, -0.4]]),
            structural_logits=torch.tensor([[0.5, 2.4, -0.3], [0.4, 2.1, -0.2]]),
            destroy_logits=torch.tensor([[2.2, 0.8, -0.2], [1.9, 0.6, -0.1]]),
        )

        _, _, _, _, _, rewards, aux = adapter._compute_group_signals(x, raw_logits)
        expected = sum(
            weights[name] * aux[key]
            for name, key in (
                ("target", "target_prob_rewards"),
                ("entropy", "entropy_rewards"),
                ("source", "source_agreement_rewards"),
                ("consensus", "consensus_rewards"),
            )
        )

        self.assertTrue(torch.allclose(rewards, expected))

    def test_eq7_reward_weights_reject_negative_values(self):
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            self._make_gn_adapter(
                reward_config={
                    "aggregation": "legacy_weighted",
                    "legacy_weights": {"target": 0.55, "entropy": -0.05, "source": 0.30, "consensus": 0.20},
                }
            )

    def test_eq7_reward_weights_reject_non_unit_sum(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            self._make_gn_adapter(
                reward_config={
                    "aggregation": "legacy_weighted",
                    "legacy_weights": {"target": 0.45, "entropy": 0.20, "source": 0.20, "consensus": 0.20},
                }
            )

    def test_gn_raw_only_view_set_zeroes_optional_group_signals(self):
        x = torch.randn(2, 4, 4, 4)
        raw_logits = torch.tensor(
            [[4.0, 0.5, -0.5], [3.0, 0.3, -0.2]],
            dtype=torch.float32,
        )
        source_logits = torch.tensor(
            [[3.8, 0.6, -0.4], [2.9, 0.4, -0.1]],
            dtype=torch.float32,
        )
        weak_logits = torch.tensor(
            [[0.5, 4.0, -0.5], [0.4, 3.5, -0.2]],
            dtype=torch.float32,
        )
        structural_logits = torch.tensor(
            [[-0.5, 0.5, 4.2], [-0.3, 0.6, 3.8]],
            dtype=torch.float32,
        )
        destroy_logits = torch.tensor(
            [[0.2, 3.8, -0.6], [0.1, 3.2, -0.2]],
            dtype=torch.float32,
        )

        adapter = self._make_gn_adapter(active_views=["raw"])
        self._patch_group_signal_views(
            adapter,
            x,
            source_logits=source_logits,
            weak_logits=weak_logits,
            structural_logits=structural_logits,
            destroy_logits=destroy_logits,
        )
        _, _, _, _, _, rewards, aux_info = adapter._compute_group_signals(x, raw_logits)

        self.assertEqual(rewards.shape[1], 1)
        self.assertTrue(torch.allclose(aux_info["destroy_sensitivity"], torch.zeros_like(aux_info["destroy_sensitivity"])))
        self.assertTrue(torch.allclose(aux_info["view_inconsistency"], torch.zeros_like(aux_info["view_inconsistency"])))

    def test_ln_deyo_param_scope_filters_to_first_nine_block_norms(self):
        adapter = self._make_ln_adapter(enabled=True, param_scope="deyo_vit")
        params = [nn.Parameter(torch.tensor(float(idx))) for idx in range(6)]
        names = [
            "blocks.0.norm1.weight",
            "blocks.8.norm2.bias",
            "blocks.9.norm1.weight",
            "blocks.11.norm2.bias",
            "norm.weight",
            "head.weight",
        ]

        filtered_params, filtered_names = adapter._filter_ln_params_by_scope(params, names)

        self.assertEqual(filtered_names, names[:2])
        self.assertIs(filtered_params[0], params[0])
        self.assertIs(filtered_params[1], params[1])

    def test_ln_natural_shift_bridge_uses_shared_deyo_chain_and_weighted_updates(self):
        adapter = self._make_ln_adapter(
            enabled=True,
            param_scope="deyo_vit",
            vit_ln_overrides={"source_calib_max_steps": 8},
        )
        adapter._ensure_ln_state(torch.device("cpu"), torch.float32)
        x = torch.randn(4, 1, 16, 16)
        raw_logits = torch.tensor(
            [
                [5.0, 0.0, -1.0],
                [0.0, 5.0, -1.0],
                [1.1, 1.0, 0.9],
                [4.5, 0.2, -1.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        plpd_logits = torch.tensor(
            [
                [1.0, 0.8, 0.6],
                [0.6, 0.9, 0.7],
                [1.0, 1.0, 1.0],
                [4.3, 0.2, -1.0],
            ],
            dtype=torch.float32,
        )
        source_logits = torch.tensor(
            [
                [4.8, 0.2, -0.8],
                [0.2, 4.7, -0.8],
                [1.1, 1.0, 0.8],
                [4.2, 0.3, -0.7],
            ],
            dtype=torch.float32,
        )

        def fake_model_forward(inp):
            if inp.data_ptr() == x.data_ptr():
                return raw_logits
            return plpd_logits

        adapter.model.forward = fake_model_forward
        adapter.source_model.forward = lambda inp: source_logits
        captured = {}

        def fake_topology(current_probs, source_probs, sample_weights=None):
            captured["topology_weights"] = sample_weights.detach().clone()
            return current_probs.new_tensor(0.25), int(current_probs.shape[0])

        def fake_update(probs, sample_weights=None):
            captured["template_weights"] = sample_weights.detach().clone()

        adapter._compute_ln_topology_loss = fake_topology
        adapter._update_ln_class_template = fake_update

        updated = adapter._adapt_on_batch_ln_vit(x)

        raw_probs = raw_logits.softmax(dim=1)
        raw_entropy = softmax_entropy(raw_logits)
        plpd_probs = plpd_logits.softmax(dim=1)
        top1 = raw_probs.argmax(dim=1, keepdim=True)
        plpd_scores = (raw_probs.gather(1, top1) - plpd_probs.gather(1, top1)).reshape(-1)
        entropy_mask = raw_entropy < (0.5 * torch.log(torch.tensor(3.0)))
        final_mask = entropy_mask & (plpd_scores > 0.2)
        expected_coeff = adapter._compute_ln_deyo_coeff(
            raw_entropy[final_mask],
            plpd_scores[final_mask],
        )

        self.assertTrue(updated)
        self.assertEqual(adapter.last_ln_natural_shift_profile, "shared")
        self.assertEqual(adapter.last_ln_selection_mode, "entropy_plpd_deyo_bridge")
        self.assertFalse(adapter.last_ln_entropy_calibration_active)
        self.assertEqual(adapter.last_ln_param_scope, "deyo_vit")
        self.assertAlmostEqual(
            adapter.last_ln_selected_ratio,
            float(final_mask.float().mean().item()),
            places=6,
        )
        self.assertTrue(torch.allclose(captured["topology_weights"], expected_coeff))
        self.assertTrue(torch.allclose(captured["template_weights"], expected_coeff))
        self.assertAlmostEqual(adapter.last_ln_deyo_coeff_mean, float(expected_coeff.mean().item()), places=6)
        self.assertAlmostEqual(adapter.last_ln_deyo_coeff_max, float(expected_coeff.max().item()), places=6)

    def test_ln_aggregate_stats_report_run_level_means(self):
        adapter = self._make_ln_adapter(enabled=True)
        adapter.last_ln_entropy_keep_ratio = 0.50
        adapter.last_ln_plpd_keep_ratio = 0.25
        adapter.last_ln_selected_ratio = 0.20
        adapter.last_ln_natural_shift_rescue_active = True
        adapter.last_ln_entropy_active_classes = 11
        adapter.last_ln_deyo_coeff_mean = 2.0
        adapter.last_ln_deyo_coeff_max = 3.0
        adapter.last_ln_active_classes = 5
        adapter._record_ln_aggregate_stats()

        adapter.last_ln_entropy_keep_ratio = 1.00
        adapter.last_ln_plpd_keep_ratio = 0.75
        adapter.last_ln_selected_ratio = 0.60
        adapter.last_ln_natural_shift_rescue_active = False
        adapter.last_ln_entropy_active_classes = 3
        adapter.last_ln_deyo_coeff_mean = 1.0
        adapter.last_ln_deyo_coeff_max = 1.5
        adapter.last_ln_active_classes = 7
        adapter._record_ln_aggregate_stats()

        stats = adapter.get_stats()
        self.assertAlmostEqual(stats["mean_ln_entropy_keep_ratio"], 0.75, places=6)
        self.assertAlmostEqual(stats["mean_ln_plpd_keep_ratio"], 0.50, places=6)
        self.assertAlmostEqual(stats["mean_ln_selected_ratio"], 0.40, places=6)
        self.assertAlmostEqual(stats["mean_ln_sketch_rescue_step_ratio"], 0.50, places=6)
        self.assertAlmostEqual(stats["mean_ln_sketch_rescue_active_classes"], 11.0, places=6)
        self.assertAlmostEqual(stats["mean_ln_deyo_coeff_mean"], 1.5, places=6)
        self.assertAlmostEqual(stats["mean_ln_deyo_coeff_max"], 2.25, places=6)
        self.assertAlmostEqual(stats["mean_ln_selected_active_classes"], 6.0, places=6)

    def test_ln_probe_policy_alternate_and_off_are_stateful_without_cache(self):
        adapter = self._make_ln_adapter(
            vit_ln_overrides={"probe_policy": "alternate"}
        )
        self.assertTrue(adapter._should_run_ln_probe())
        adapter.step_count = 1
        self.assertFalse(adapter._should_run_ln_probe())
        self.assertEqual(adapter.ln_probe_forward_count, 1)
        self.assertEqual(adapter.ln_probe_skipped_count, 1)
        adapter_off = self._make_ln_adapter(vit_ln_overrides={"probe_policy": "off"})
        self.assertFalse(adapter_off._should_run_ln_probe())
        self.assertEqual(adapter_off.ln_probe_forward_count, 0)

    def test_group_diagnostic_exports_eq7_terms(self):
        adapter = self._make_gn_adapter()
        diagnostic = adapter.diagnose_group_signals(torch.randn(2, 4, 4, 4))
        for key in (
            "rewards", "reward_std", "entropy_rewards", "source_agreement_rewards",
            "consensus_rewards", "destroy_sensitivity", "safety_score", "safety_weight",
            "selected", "raw_view", "photometric_view", "structural_view", "destroy_view",
        ):
            self.assertIn(key, diagnostic)
        self.assertEqual(diagnostic["rewards"].shape[0], 2)
        self.assertTrue(torch.isfinite(diagnostic["safety_weight"]).all())

    def test_ln_disabled_natural_shift_bridge_falls_back_to_legacy_continual_path(self):
        adapter = self._make_ln_adapter(
            enabled=False,
            vit_ln_overrides={"source_calib_max_steps": 0},
        )
        adapter._ensure_ln_state(torch.device("cpu"), torch.float32)
        x = torch.randn(3, 1, 16, 16)
        raw_logits = torch.tensor(
            [
                [5.0, 0.0, -1.0],
                [0.0, 5.0, -1.0],
                [-1.0, 0.0, 5.0],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        plpd_logits = torch.tensor(
            [
                [2.0, 1.5, 1.0],
                [1.2, 2.1, 0.8],
                [0.8, 1.2, 2.0],
            ],
            dtype=torch.float32,
        )
        source_logits = torch.tensor(
            [
                [4.7, 0.3, -0.7],
                [0.3, 4.8, -0.7],
                [-0.7, 0.3, 4.8],
            ],
            dtype=torch.float32,
        )

        def fake_model_forward(inp):
            if inp.data_ptr() == x.data_ptr():
                return raw_logits
            return plpd_logits

        adapter.model.forward = fake_model_forward
        adapter.source_model.forward = lambda inp: source_logits

        updated = adapter._adapt_on_batch_ln_vit(x)

        self.assertTrue(updated)
        self.assertEqual(adapter.last_ln_natural_shift_profile, "none")
        self.assertEqual(adapter.last_ln_selection_mode, "entropy_plpd")
        self.assertEqual(adapter.last_ln_loss_mode, "adadem_topology")
        self.assertFalse(adapter.last_ln_entropy_calibration_active)
        self.assertEqual(adapter.last_ln_deyo_coeff_mean, 0.0)


if __name__ == "__main__":
    unittest.main()
