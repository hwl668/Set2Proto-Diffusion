from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from set2proto.config import ConfigError, load_config, validate_config


CONFIG_PATH = PROJECT_ROOT / "configs" / "mvp.yaml"


class ConfigTests(unittest.TestCase):
    def test_smoke_profile_deep_merges_defaults(self) -> None:
        resolved = load_config(CONFIG_PATH, "smoke")
        config = resolved.values

        self.assertEqual(resolved.profile, "smoke")
        self.assertEqual(config["training"]["batch_size"], 2)
        self.assertEqual(config["training"]["max_steps"], 2)
        self.assertEqual(config["model"]["hidden_dim"], 256)
        self.assertEqual(config["quantization"]["codebook_size"], 1024)
        self.assertEqual(config["runtime"]["profile"], "smoke")

    def test_paths_are_absolute_and_root_relative(self) -> None:
        resolved = load_config(CONFIG_PATH, "full")
        for value in resolved.values["paths"].values():
            self.assertTrue(Path(value).is_absolute())
        self.assertEqual(
            Path(resolved.values["paths"]["output_dir"]),
            PROJECT_ROOT / "outputs",
        )

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown profile"):
            load_config(CONFIG_PATH, "does-not-exist")

    def test_hidden_dimension_must_be_divisible_by_heads(self) -> None:
        config = load_config(CONFIG_PATH, "full").to_dict()
        config["model"]["hidden_dim"] = 250
        with self.assertRaisesRegex(ConfigError, "must be divisible"):
            validate_config(config)

    def test_non_positive_batch_size_is_rejected(self) -> None:
        config = load_config(CONFIG_PATH, "full").to_dict()
        config["training"]["batch_size"] = 0
        with self.assertRaisesRegex(ConfigError, "batch_size"):
            validate_config(config)

    def test_memory_limit_above_project_cap_is_rejected(self) -> None:
        config = load_config(CONFIG_PATH, "full").to_dict()
        config["runtime"]["max_peak_memory_gib"] = 7.21
        with self.assertRaisesRegex(ConfigError, "max_peak_memory_gib"):
            validate_config(config)

    def test_p1_1_training_protocol_rejects_invalid_values(self) -> None:
        cases = (
            ("oracle_method", "teacher_convex_residual_oracle", "oracle_method"),
            ("hidden_dim", 0, ">="),
            ("dropout", 1.1, "<="),
            ("max_parameters", 0, ">="),
            ("batch_size", 0, ">="),
            ("precision", "fp32", "precision"),
            ("selection_min_hard_auc_gain", -0.1, ">="),
        )
        for key, value, message in cases:
            with self.subTest(key=key):
                config = load_config(CONFIG_PATH, "pilot").to_dict()
                config["p1_1"][key] = value
                with self.assertRaisesRegex(ConfigError, message):
                    validate_config(config)

    def test_p1_1_min_steps_cannot_exceed_max_steps(self) -> None:
        config = load_config(CONFIG_PATH, "pilot").to_dict()
        config["p1_1"]["min_steps"] = 2001
        with self.assertRaisesRegex(ConfigError, "cannot exceed"):
            validate_config(config)

    def test_p1_2_protocol_rejects_invalid_gallery_values(self) -> None:
        cases = (
            ("gallery_temperature", 0.0, ">="),
            ("gallery_temperature", 1.1, "<="),
            ("gallery_margin", -0.1, ">="),
            ("gate_hidden_dim", 0, ">="),
            ("precision", "fp32", "precision"),
        )
        for key, value, message in cases:
            with self.subTest(key=key):
                config = load_config(CONFIG_PATH, "pilot").to_dict()
                config["p1_2"][key] = value
                with self.assertRaisesRegex(ConfigError, message):
                    validate_config(config)

    def test_p1_2_recipes_must_match_and_enable_identity_losses(self) -> None:
        config = load_config(CONFIG_PATH, "pilot").to_dict()
        del config["p1_2"]["recipes"]["anchor_safe"]
        with self.assertRaisesRegex(ConfigError, "exactly match"):
            validate_config(config)
        config = load_config(CONFIG_PATH, "pilot").to_dict()
        config["p1_2"]["recipes"]["anchor_safe"][
            "gallery_margin_weight"
        ] = 0.0
        with self.assertRaisesRegex(ConfigError, "both direct"):
            validate_config(config)

    def test_p1_3_gate_and_relative_margin_protocol_is_validated(self) -> None:
        cases = (
            ("gate_initial_value", 0.0, ">="),
            ("maximum_gate", 1.1, "<="),
            ("gallery_temperature", 0.0, ">="),
            ("precision", "fp32", "precision"),
        )
        for key, value, message in cases:
            with self.subTest(key=key):
                config = load_config(CONFIG_PATH, "pilot").to_dict()
                config["p1_3"][key] = value
                with self.assertRaisesRegex(ConfigError, message):
                    validate_config(config)
        config = load_config(CONFIG_PATH, "pilot").to_dict()
        config["p1_3"]["gate_initial_value"] = 0.4
        with self.assertRaisesRegex(ConfigError, "less than"):
            validate_config(config)

    def test_p1_3_requires_anchor_relative_losses(self) -> None:
        config = load_config(CONFIG_PATH, "pilot").to_dict()
        config["p1_3"]["recipes"]["scalar_frozen"][
            "relative_gallery_margin_weight"
        ] = 0.0
        with self.assertRaisesRegex(ConfigError, "relative margin"):
            validate_config(config)

    def test_synthetic_images_must_cover_disjoint_condition_and_teacher(self) -> None:
        config = load_config(CONFIG_PATH, "full").to_dict()
        config["data"]["synthetic"]["images_per_identity"] = 7
        with self.assertRaisesRegex(ConfigError, "condition_frames"):
            validate_config(config)

    def test_real_images_must_cover_disjoint_condition_and_teacher(self) -> None:
        config = load_config(CONFIG_PATH, "full").to_dict()
        config["data"]["real"]["min_images_per_identity"] = 7
        with self.assertRaisesRegex(ConfigError, "data.real"):
            validate_config(config)

    def test_codebook_cannot_exceed_training_teacher_tokens(self) -> None:
        config = load_config(CONFIG_PATH, "smoke").to_dict()
        config["quantization"]["codebook_size"] = 100000
        with self.assertRaisesRegex(ConfigError, "training teacher tokens"):
            validate_config(config)

    def test_fallback_must_start_at_active_batch_size(self) -> None:
        config = load_config(CONFIG_PATH, "pilot").to_dict()
        config["runtime"]["oom_batch_fallback"] = [8, 4, 2, 1]
        with self.assertRaisesRegex(ConfigError, "must start"):
            validate_config(config)

    def test_p0_2_invalid_search_and_calibration_values_are_rejected(self) -> None:
        cases = (
            ("reference_top_k_candidates", [0], "positive integers"),
            ("reference_top_k_candidates", [True], "positive integers"),
            ("identity_weight_candidates", [-0.5], "non-negative"),
            ("minimum_frame_weight", 0.25, "less than"),
            ("clean_gap_quantile", 1.0, r"must be in \(0, 1\)"),
            ("reliability_scale_floor", 0.0, ">="),
            ("identity_scale_floor", 0.0, ">="),
            ("local_z_clip", [1.0, 1.0], "increasing pair"),
            ("selection_clean_auc_max_drop", -0.1, ">="),
            ("selection_rank1_max_drop", -0.1, ">="),
        )
        for key, value, message in cases:
            with self.subTest(key=key, value=value):
                config = load_config(CONFIG_PATH, "pilot").to_dict()
                config["p0_2"][key] = value
                with self.assertRaisesRegex(ConfigError, message):
                    validate_config(config)

    def test_p0_3_hook_protocol_rejects_invalid_values(self) -> None:
        cases = (
            ("candidate_hooks", ["body.3", "body.3"], "unique"),
            ("candidate_hooks", ["output_layer.1"], r"body\.<index>"),
            ("visibility_resize", "bilinear", "nearest"),
            ("cache_dtype", "float32", "float16"),
            ("set_batch_size", 0, ">="),
            ("selection_min_hard_auc_gain", -0.1, ">="),
            ("selection_min_hard_scenario_wins", 0, ">="),
            ("suffix_replay_min_map_cosine", 0.0, "positive"),
            ("tie_break_prefer_later", 1, "boolean"),
        )
        for key, value, message in cases:
            with self.subTest(key=key, value=value):
                config = load_config(CONFIG_PATH, "pilot").to_dict()
                config["p0_3"][key] = value
                if key == "candidate_hooks":
                    config["p0_3"]["expected_shapes"] = {
                        name: [128, 28, 28] for name in value
                    }
                with self.assertRaisesRegex(ConfigError, message):
                    validate_config(config)

    def test_p1_0_residual_oracle_protocol_rejects_invalid_values(self) -> None:
        config = load_config(CONFIG_PATH, "pilot").to_dict()
        config["p1_0"]["candidate_methods"] = ["unknown_oracle"]
        with self.assertRaises(ConfigError):
            validate_config(config)

        config = load_config(CONFIG_PATH, "pilot").to_dict()
        config["p1_0"]["span_residual_norm_cap"] = 0.0
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_to_dict_is_a_defensive_copy(self) -> None:
        resolved = load_config(CONFIG_PATH, "smoke")
        copied = resolved.to_dict()
        original = copy.deepcopy(resolved.values)
        copied["model"]["hidden_dim"] = 64
        self.assertEqual(resolved.values, original)


if __name__ == "__main__":
    unittest.main()
