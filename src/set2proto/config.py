"""Configuration loading, profile merging, and MVP invariant validation."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when an MVP configuration is missing or internally inconsistent."""


@dataclass(frozen=True)
class ResolvedConfig:
    """A validated configuration with all workspace paths made absolute."""

    values: dict[str, Any]
    profile: str
    config_path: Path
    project_root: Path

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.values)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{key}' must be a mapping")
    return value


def _require_int(
    mapping: Mapping[str, Any],
    key: str,
    *,
    minimum: int = 1,
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ConfigError(f"'{key}' must be an integer")
    value = int(value)
    if value < minimum:
        raise ConfigError(f"'{key}' must be >= {minimum}, got {value}")
    return value


def _require_real(
    mapping: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigError(f"'{key}' must be numeric")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ConfigError(f"'{key}' must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"'{key}' must be <= {maximum}, got {value}")
    return value


def _require_bool(mapping: Mapping[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"'{key}' must be boolean")
    return value


def _validate_positive_int_sequence(
    values: Any,
    *,
    name: str,
    expected_first: int | None = None,
) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ConfigError(f"'{name}' must be a non-empty list")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ConfigError(f"'{name}' must contain positive integers")
        result.append(int(value))
    if any(left < right for left, right in zip(result, result[1:])):
        raise ConfigError(f"'{name}' must be in non-increasing fallback order")
    if expected_first is not None and result[0] != expected_first:
        raise ConfigError(
            f"'{name}' must start at the configured value {expected_first}"
        )
    return result


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the invariants required by every MVP profile."""

    required_sections = (
        "project",
        "paths",
        "data",
        "quantization",
        "projection",
        "teacher",
        "p0_2",
        "p0_3",
        "p2_1",
        "backbone",
        "model",
        "training",
        "decoding",
        "evaluation",
        "runtime",
        "logging",
    )
    for section in required_sections:
        _require_mapping(config, section)

    project = _require_mapping(config, "project")
    if not isinstance(project.get("name"), str) or not project["name"].strip():
        raise ConfigError("'project.name' must be a non-empty string")
    _require_int(project, "seed", minimum=0)

    paths = _require_mapping(config, "paths")
    for key in ("data_dir", "weights_dir", "cache_dir", "output_dir"):
        if not isinstance(paths.get(key), str) or not paths[key].strip():
            raise ConfigError(f"'paths.{key}' must be a non-empty path string")

    data = _require_mapping(config, "data")
    condition_frames = _require_int(data, "condition_frames")
    _require_int(data, "teacher_frames")
    _require_int(data, "input_size")
    feature_grid = _require_mapping(data, "feature_grid")
    _require_int(feature_grid, "height")
    _require_int(feature_grid, "width")
    _require_int(feature_grid, "backbone_dim")
    _require_int(feature_grid, "projected_dim")
    synthetic = _require_mapping(data, "synthetic")
    for key in ("train_identities", "val_identities", "test_identities"):
        _require_int(synthetic, key, minimum=2)
    for key in ("images_per_identity", "sets_per_identity"):
        _require_int(synthetic, key)
    minimum_images = condition_frames + int(data["teacher_frames"])
    if int(synthetic["images_per_identity"]) < minimum_images:
        raise ConfigError(
            "'data.synthetic.images_per_identity' must be at least "
            "condition_frames + teacher_frames "
            f"({minimum_images})"
        )
    real = _require_mapping(data, "real")
    if real.get("dataset") != "celeba":
        raise ConfigError("'data.real.dataset' must be 'celeba' for MVP")
    for key in ("train_identities", "val_identities", "test_identities"):
        _require_int(real, key, minimum=2)
    real_minimum_images = _require_int(
        real,
        "min_images_per_identity",
    )
    if real_minimum_images < minimum_images:
        raise ConfigError(
            "'data.real.min_images_per_identity' must be at least "
            "condition_frames + teacher_frames "
            f"({minimum_images})"
        )
    _require_int(real, "expected_images")
    _require_int(real, "expected_identities")
    sets_per_identity = _require_int(real, "sets_per_identity")
    _require_int(real, "set_max_identities_per_split", minimum=0)
    if sets_per_identity < 5:
        raise ConfigError(
            "'data.real.sets_per_identity' must cover all five MVP scenarios"
        )
    corruptions = _require_mapping(data, "corruptions")
    for key in (
        "gaussian_blur",
        "motion_blur",
        "low_light",
        "jpeg",
        "random_occlusion",
        "complementary_occlusion",
        "common_occlusion",
    ):
        _require_bool(corruptions, key)
    _require_real(
        corruptions,
        "wrong_identity_probability",
        minimum=0.0,
        maximum=1.0,
    )

    quantization = _require_mapping(config, "quantization")
    if quantization.get("method") != "spherical_kmeans":
        raise ConfigError("'quantization.method' must be 'spherical_kmeans' for MVP")
    _require_int(quantization, "codebook_size", minimum=2)
    _require_int(quantization, "max_fit_tokens")
    _require_int(quantization, "batch_size")
    _require_int(quantization, "iterations")
    _require_int(quantization, "n_init")
    _require_real(
        quantization,
        "gate_min_cosine",
        minimum=-1.0,
        maximum=1.0,
    )
    _require_real(
        quantization,
        "gate_min_utilization",
        minimum=0.0,
        maximum=1.0,
    )
    available_teacher_tokens = (
        int(synthetic["train_identities"])
        * int(synthetic["sets_per_identity"])
        * int(feature_grid["height"])
        * int(feature_grid["width"])
    )
    if int(quantization["codebook_size"]) > available_teacher_tokens:
        raise ConfigError(
            "'quantization.codebook_size' cannot exceed the number of "
            f"training teacher tokens ({available_teacher_tokens})"
        )

    projection = _require_mapping(config, "projection")
    if projection.get("method") != "pca":
        raise ConfigError("'projection.method' must be 'pca' for MVP")
    projection_fit_tokens = _require_int(projection, "max_fit_tokens")
    if projection_fit_tokens < int(feature_grid["projected_dim"]):
        raise ConfigError(
            "'projection.max_fit_tokens' must be at least projected_dim"
        )
    if projection.get("svd_solver") not in {"randomized", "full"}:
        raise ConfigError(
            "'projection.svd_solver' must be 'randomized' or 'full'"
        )

    teacher = _require_mapping(config, "teacher")
    if teacher.get("mode") not in {"disjoint", "self"}:
        raise ConfigError("'teacher.mode' must be 'disjoint' or 'self'")
    _require_real(
        teacher,
        "robust_keep_fraction",
        minimum=0.01,
        maximum=1.0,
    )
    consensus_weight = _require_real(
        teacher,
        "consensus_weight",
        minimum=0.0,
    )
    quality_weight = _require_real(
        teacher,
        "quality_weight",
        minimum=0.0,
    )
    if consensus_weight == 0.0 and quality_weight == 0.0:
        raise ConfigError(
            "at least one teacher consensus/quality weight must be positive"
        )

    p0_2 = _require_mapping(config, "p0_2")
    top_k_candidates = p0_2.get("reference_top_k_candidates")
    if (
        not isinstance(top_k_candidates, list)
        or not top_k_candidates
        or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value <= 0
            for value in top_k_candidates
        )
    ):
        raise ConfigError(
            "'p0_2.reference_top_k_candidates' must contain positive integers"
        )
    identity_weights = p0_2.get("identity_weight_candidates")
    if (
        not isinstance(identity_weights, list)
        or not identity_weights
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or value < 0
            for value in identity_weights
        )
    ):
        raise ConfigError(
            "'p0_2.identity_weight_candidates' must contain non-negative "
            "numbers"
        )
    _require_real(p0_2, "local_reliability_weight", minimum=0.0)
    _require_real(p0_2, "quality_weight", minimum=0.0)
    minimum_frame_weight = _require_real(
        p0_2,
        "minimum_frame_weight",
        minimum=0.0,
    )
    if minimum_frame_weight >= 1.0 / float(condition_frames):
        raise ConfigError(
            "'p0_2.minimum_frame_weight' must be less than 1/condition_frames"
        )
    clean_gap_quantile = _require_real(
        p0_2,
        "clean_gap_quantile",
        minimum=0.0,
        maximum=1.0,
    )
    if clean_gap_quantile in {0.0, 1.0}:
        raise ConfigError("'p0_2.clean_gap_quantile' must be in (0, 1)")
    _require_real(p0_2, "reliability_scale_floor", minimum=1e-8)
    _require_real(p0_2, "identity_scale_floor", minimum=1e-8)
    local_z_clip = p0_2.get("local_z_clip")
    if (
        not isinstance(local_z_clip, list)
        or len(local_z_clip) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in local_z_clip
        )
        or float(local_z_clip[0]) >= float(local_z_clip[1])
    ):
        raise ConfigError(
            "'p0_2.local_z_clip' must be an increasing pair"
        )
    _require_real(
        p0_2,
        "selection_clean_auc_max_drop",
        minimum=0.0,
    )
    _require_real(
        p0_2,
        "selection_rank1_max_drop",
        minimum=0.0,
    )

    p0_3 = _require_mapping(config, "p0_3")
    candidate_hooks = p0_3.get("candidate_hooks")
    if (
        not isinstance(candidate_hooks, list)
        or not candidate_hooks
        or len(set(candidate_hooks)) != len(candidate_hooks)
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"body\.\d+", value) is None
            for value in candidate_hooks
        )
    ):
        raise ConfigError(
            "'p0_3.candidate_hooks' must contain unique body.<index> names"
        )
    expected_shapes = p0_3.get("expected_shapes")
    if (
        not isinstance(expected_shapes, Mapping)
        or set(expected_shapes) != set(candidate_hooks)
    ):
        raise ConfigError(
            "'p0_3.expected_shapes' keys must match candidate_hooks"
        )
    for hook_name, shape in expected_shapes.items():
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value <= 0
                for value in shape
            )
        ):
            raise ConfigError(
                f"'p0_3.expected_shapes.{hook_name}' must be [C,H,W]"
            )
    if p0_3.get("visibility_resize") != "nearest":
        raise ConfigError("'p0_3.visibility_resize' must be 'nearest'")
    if p0_3.get("cache_dtype") != "float16":
        raise ConfigError("'p0_3.cache_dtype' must be 'float16'")
    _require_int(p0_3, "set_batch_size")
    _require_real(p0_3, "selection_min_hard_auc_gain", minimum=0.0)
    _require_real(
        p0_3,
        "selection_min_complementary_auc_gain",
        minimum=0.0,
    )
    _require_int(p0_3, "selection_min_hard_scenario_wins", minimum=1)
    _require_real(p0_3, "selection_clean_auc_max_drop", minimum=0.0)
    _require_real(p0_3, "selection_rank1_max_drop", minimum=0.0)
    replay_cosine = _require_real(
        p0_3,
        "suffix_replay_min_map_cosine",
        minimum=0.0,
        maximum=1.0,
    )
    if replay_cosine <= 0.0:
        raise ConfigError(
            "'p0_3.suffix_replay_min_map_cosine' must be positive"
        )
    _require_bool(p0_3, "tie_break_prefer_later")

    p1_0 = _require_mapping(config, "p1_0")
    allowed_residual_methods = {
        "teacher_best_frame_oracle",
        "teacher_convex_residual_oracle",
        "teacher_bounded_span_residual_oracle",
    }
    residual_methods = p1_0.get("candidate_methods")
    if (
        not isinstance(residual_methods, list)
        or not residual_methods
        or len(set(residual_methods)) != len(residual_methods)
        or any(
            not isinstance(value, str)
            or value not in allowed_residual_methods
            for value in residual_methods
        )
    ):
        raise ConfigError(
            "'p1_0.candidate_methods' must contain unique supported "
            "continuous residual oracle names"
        )
    _require_int(p1_0, "convex_iterations")
    _require_real(p1_0, "convex_step_size", minimum=1e-8)
    _require_real(p1_0, "span_ridge", minimum=1e-8)
    _require_real(
        p1_0,
        "span_residual_norm_cap",
        minimum=1e-8,
    )
    _require_real(
        p1_0,
        "selection_min_hard_auc_gain",
        minimum=0.0,
    )
    _require_real(
        p1_0,
        "selection_min_complementary_auc_gain",
        minimum=0.0,
    )
    _require_int(
        p1_0,
        "selection_min_hard_scenario_wins",
        minimum=1,
    )
    _require_real(
        p1_0,
        "selection_clean_auc_max_drop",
        minimum=0.0,
    )
    _require_real(
        p1_0,
        "selection_rank1_max_drop",
        minimum=0.0,
    )
    _require_real(
        p1_0,
        "selection_min_teacher_map_cosine_gain",
        minimum=0.0,
    )

    p1_1 = _require_mapping(config, "p1_1")
    if p1_1.get("oracle_method") != "teacher_best_frame_oracle":
        raise ConfigError(
            "'p1_1.oracle_method' must be 'teacher_best_frame_oracle'"
        )
    _require_int(p1_1, "hidden_dim")
    _require_real(p1_1, "dropout", minimum=0.0, maximum=1.0)
    _require_int(p1_1, "max_parameters")
    _require_int(p1_1, "batch_size")
    _require_int(p1_1, "smoke_steps")
    p1_1_min_steps = _require_int(p1_1, "min_steps")
    p1_1_max_steps = _require_int(p1_1, "max_steps")
    if p1_1_min_steps > p1_1_max_steps:
        raise ConfigError("'p1_1.min_steps' cannot exceed max_steps")
    _require_int(p1_1, "validate_every_steps")
    _require_int(p1_1, "early_stop_patience")
    _require_real(p1_1, "learning_rate", minimum=1e-12)
    _require_real(p1_1, "weight_decay", minimum=0.0)
    p1_1_warmup = _require_int(p1_1, "warmup_steps", minimum=0)
    if p1_1_warmup >= p1_1_max_steps:
        raise ConfigError("'p1_1.warmup_steps' must be less than max_steps")
    _require_real(p1_1, "max_grad_norm", minimum=1e-12)
    loss_weights = [
        _require_real(p1_1, key, minimum=0.0)
        for key in (
            "distillation_weight",
            "local_map_cosine_weight",
            "global_identity_weight",
        )
    ]
    if not any(weight > 0.0 for weight in loss_weights):
        raise ConfigError("'p1_1' must enable at least one training loss")
    if p1_1.get("precision") not in {"bf16", "fp16"}:
        raise ConfigError("'p1_1.precision' must be 'bf16' or 'fp16'")
    _require_real(
        p1_1,
        "selection_min_hard_auc_gain",
        minimum=0.0,
    )
    _require_real(
        p1_1,
        "selection_min_complementary_auc_gain",
        minimum=0.0,
    )
    _require_int(
        p1_1,
        "selection_min_hard_scenario_wins",
        minimum=1,
    )
    _require_real(
        p1_1,
        "selection_clean_auc_max_drop",
        minimum=0.0,
    )
    _require_real(
        p1_1,
        "selection_rank1_max_drop",
        minimum=0.0,
    )
    _require_real(
        p1_1,
        "selection_min_teacher_map_cosine_gain",
        minimum=0.0,
    )

    p1_2 = _require_mapping(config, "p1_2")
    recipe_names = p1_2.get("candidate_recipes")
    if (
        not isinstance(recipe_names, list)
        or len(recipe_names) < 2
        or len(set(recipe_names)) != len(recipe_names)
        or any(
            not isinstance(name, str) or not name.strip()
            for name in recipe_names
        )
    ):
        raise ConfigError(
            "'p1_2.candidate_recipes' must contain at least two unique names"
        )
    recipes = _require_mapping(p1_2, "recipes")
    if set(recipes) != set(recipe_names):
        raise ConfigError(
            "'p1_2.recipes' keys must exactly match candidate_recipes"
        )
    recipe_weight_names = (
        "distillation_weight",
        "gate_target_weight",
        "local_teacher_weight",
        "global_teacher_weight",
        "gallery_cross_entropy_weight",
        "gallery_margin_weight",
        "anchor_regularization_weight",
    )
    for recipe_name in recipe_names:
        recipe = _require_mapping(recipes, recipe_name)
        weights = [
            _require_real(recipe, key, minimum=0.0)
            for key in recipe_weight_names
        ]
        if not any(weight > 0.0 for weight in weights):
            raise ConfigError(
                f"'p1_2.recipes.{recipe_name}' must enable a loss"
            )
        if (
            float(recipe["gallery_cross_entropy_weight"]) <= 0.0
            or float(recipe["gallery_margin_weight"]) <= 0.0
        ):
            raise ConfigError(
                f"'p1_2.recipes.{recipe_name}' must enable both direct "
                "gallery losses"
            )
    _require_int(p1_2, "gate_hidden_dim")
    _require_real(p1_2, "gate_initial_bias")
    _require_real(
        p1_2,
        "gate_teacher_improvement_epsilon",
        minimum=0.0,
    )
    temperature = _require_real(
        p1_2,
        "gallery_temperature",
        minimum=1e-8,
    )
    if temperature > 1.0:
        raise ConfigError("'p1_2.gallery_temperature' must be <= 1")
    _require_real(p1_2, "gallery_margin", minimum=0.0, maximum=2.0)
    _require_int(p1_2, "max_parameters")
    _require_int(p1_2, "batch_size")
    _require_int(p1_2, "smoke_steps")
    p1_2_min_steps = _require_int(p1_2, "min_steps")
    p1_2_max_steps = _require_int(p1_2, "max_steps")
    if p1_2_min_steps > p1_2_max_steps:
        raise ConfigError("'p1_2.min_steps' cannot exceed max_steps")
    _require_int(p1_2, "validate_every_steps")
    _require_int(p1_2, "early_stop_patience")
    _require_real(p1_2, "learning_rate", minimum=1e-12)
    _require_real(p1_2, "weight_decay", minimum=0.0)
    p1_2_warmup = _require_int(p1_2, "warmup_steps", minimum=0)
    if p1_2_warmup >= p1_2_max_steps:
        raise ConfigError("'p1_2.warmup_steps' must be less than max_steps")
    _require_real(p1_2, "max_grad_norm", minimum=1e-12)
    if p1_2.get("precision") not in {"bf16", "fp16"}:
        raise ConfigError("'p1_2.precision' must be 'bf16' or 'fp16'")
    for key in (
        "selection_min_hard_auc_gain",
        "selection_min_complementary_auc_gain",
        "selection_clean_auc_max_drop",
        "selection_rank1_max_drop",
        "selection_min_teacher_map_cosine_gain",
    ):
        _require_real(p1_2, key, minimum=0.0)
    _require_int(
        p1_2,
        "selection_min_hard_scenario_wins",
        minimum=1,
    )

    p1_3 = _require_mapping(config, "p1_3")
    p1_3_recipe_names = p1_3.get("candidate_recipes")
    if (
        not isinstance(p1_3_recipe_names, list)
        or len(p1_3_recipe_names) < 2
        or len(set(p1_3_recipe_names)) != len(p1_3_recipe_names)
        or any(
            not isinstance(name, str) or not name.strip()
            for name in p1_3_recipe_names
        )
    ):
        raise ConfigError(
            "'p1_3.candidate_recipes' must contain at least two unique names"
        )
    p1_3_recipes = _require_mapping(p1_3, "recipes")
    if set(p1_3_recipes) != set(p1_3_recipe_names):
        raise ConfigError(
            "'p1_3.recipes' keys must exactly match candidate_recipes"
        )
    p1_3_loss_names = (
        "distillation_weight",
        "gate_target_weight",
        "local_teacher_weight",
        "global_teacher_weight",
        "gallery_cross_entropy_weight",
        "relative_gallery_margin_weight",
        "anchor_safety_weight",
    )
    for recipe_name in p1_3_recipe_names:
        recipe = _require_mapping(p1_3_recipes, recipe_name)
        _require_real(
            recipe,
            "router_learning_rate_scale",
            minimum=0.0,
            maximum=1.0,
        )
        weights = [
            _require_real(recipe, key, minimum=0.0)
            for key in p1_3_loss_names
        ]
        if not any(weight > 0.0 for weight in weights):
            raise ConfigError(
                f"'p1_3.recipes.{recipe_name}' must enable a loss"
            )
        if (
            float(recipe["relative_gallery_margin_weight"]) <= 0.0
            or float(recipe["anchor_safety_weight"]) <= 0.0
        ):
            raise ConfigError(
                f"'p1_3.recipes.{recipe_name}' must enable relative margin "
                "and anchor safety"
            )
    _require_int(p1_3, "gate_hidden_dim")
    initial_gate = _require_real(
        p1_3,
        "gate_initial_value",
        minimum=1e-8,
        maximum=1.0,
    )
    maximum_gate = _require_real(
        p1_3,
        "maximum_gate",
        minimum=1e-8,
        maximum=1.0,
    )
    if initial_gate >= maximum_gate:
        raise ConfigError(
            "'p1_3.gate_initial_value' must be less than maximum_gate"
        )
    _require_real(
        p1_3,
        "gate_target_improvement_scale",
        minimum=1e-8,
    )
    _require_real(
        p1_3,
        "relative_gallery_margin_target",
        minimum=0.0,
        maximum=2.0,
    )
    p1_3_temperature = _require_real(
        p1_3,
        "gallery_temperature",
        minimum=1e-8,
    )
    if p1_3_temperature > 1.0:
        raise ConfigError("'p1_3.gallery_temperature' must be <= 1")
    _require_int(p1_3, "max_parameters")
    _require_int(p1_3, "batch_size")
    _require_int(p1_3, "smoke_steps")
    p1_3_min_steps = _require_int(p1_3, "min_steps")
    p1_3_max_steps = _require_int(p1_3, "max_steps")
    if p1_3_min_steps > p1_3_max_steps:
        raise ConfigError("'p1_3.min_steps' cannot exceed max_steps")
    _require_int(p1_3, "validate_every_steps")
    _require_int(p1_3, "early_stop_patience")
    _require_real(p1_3, "learning_rate", minimum=1e-12)
    _require_real(p1_3, "weight_decay", minimum=0.0)
    p1_3_warmup = _require_int(p1_3, "warmup_steps", minimum=0)
    if p1_3_warmup >= p1_3_max_steps:
        raise ConfigError("'p1_3.warmup_steps' must be less than max_steps")
    _require_real(p1_3, "max_grad_norm", minimum=1e-12)
    if p1_3.get("precision") not in {"bf16", "fp16"}:
        raise ConfigError("'p1_3.precision' must be 'bf16' or 'fp16'")
    for key in (
        "selection_min_hard_auc_gain",
        "selection_min_complementary_auc_gain",
        "selection_clean_auc_max_drop",
        "selection_rank1_max_drop",
        "selection_min_teacher_map_cosine_gain",
    ):
        _require_real(p1_3, key, minimum=0.0)
    _require_int(
        p1_3,
        "selection_min_hard_scenario_wins",
        minimum=1,
    )

    p2_1 = _require_mapping(config, "p2_1")
    if p2_1.get("protocol") != "validation_only":
        raise ConfigError("'p2_1.protocol' must be 'validation_only'")
    if p2_1.get("construct_test_artifacts") is not False:
        raise ConfigError("'p2_1.construct_test_artifacts' must be false")
    if p2_1.get("anchor") != "quality_pooling":
        raise ConfigError("'p2_1.anchor' must be 'quality_pooling'")
    if p2_1.get("residual_definition") != "raw_teacher_minus_anchor":
        raise ConfigError(
            "'p2_1.residual_definition' must be "
            "'raw_teacher_minus_anchor'"
        )
    if p2_1.get("codebook_method") != "euclidean_minibatch_kmeans":
        raise ConfigError(
            "'p2_1.codebook_method' must be "
            "'euclidean_minibatch_kmeans'"
        )
    for key in (
        "codebook_size",
        "max_fit_tokens",
        "batch_size",
        "iterations",
        "n_init",
        "assignment_chunk_size",
        "smoke_steps",
        "formal_steps",
        "checkpoint_every_steps",
    ):
        _require_int(p2_1, key)
    if int(p2_1["smoke_steps"]) >= int(p2_1["formal_steps"]):
        raise ConfigError("'p2_1.smoke_steps' must be less than formal_steps")
    if p2_1.get("fit_device") not in {"auto", "cpu", "cuda"}:
        raise ConfigError("'p2_1.fit_device' must be auto, cpu, or cuda")
    p2_evidence = _require_mapping(p2_1, "evidence")
    if p2_evidence.get("primary_mode") != "logits":
        raise ConfigError("'p2_1.evidence.primary_mode' must be 'logits'")
    p2_top_k = _require_int(p2_evidence, "top_k_frames")
    if p2_top_k > condition_frames:
        raise ConfigError(
            "'p2_1.evidence.top_k_frames' cannot exceed condition_frames"
        )
    _require_real(p2_evidence, "lambda", minimum=0.0)
    _require_real(
        p2_evidence,
        "identity_neighbor_fraction",
        minimum=1e-8,
        maximum=1.0,
    )
    _require_real(p2_evidence, "identity_temperature", minimum=0.0)
    _require_real(p2_evidence, "quality_weight", minimum=0.0)
    _require_real(p2_evidence, "local_temperature", minimum=0.0)
    _require_real(p2_evidence, "reliability_floor", minimum=1e-12)
    p2_gates = _require_mapping(p2_1, "gates")
    _require_real(
        p2_gates,
        "min_map_cosine_gain_over_absolute",
        minimum=0.0,
    )
    _require_real(
        p2_gates,
        "min_codebook_utilization",
        minimum=0.0,
        maximum=1.0,
    )
    for key in (
        "max_hard_auc_drop_vs_quality_anchor",
        "min_maskgit_hard_auc_gain_over_one_shot",
        "min_two_level_hard_auc_gain_over_confidence",
    ):
        _require_real(p2_gates, key, minimum=0.0)

    backbone = _require_mapping(config, "backbone")
    if backbone.get("name") != "adaface_ir50":
        raise ConfigError("'backbone.name' must be 'adaface_ir50' for MVP")
    if backbone.get("architecture") != "ir_50":
        raise ConfigError("'backbone.architecture' must be 'ir_50' for MVP")
    for key in (
        "checkpoint_filename",
        "checkpoint_source",
        "checkpoint_url",
        "input_color_order",
        "spatial_hook",
        "cache_dtype",
    ):
        if not isinstance(backbone.get(key), str) or not backbone[key].strip():
            raise ConfigError(f"'backbone.{key}' must be a non-empty string")
    if backbone["input_color_order"] != "bgr":
        raise ConfigError("'backbone.input_color_order' must be 'bgr'")
    if backbone["spatial_hook"] != "auto":
        raise ConfigError("'backbone.spatial_hook' must be 'auto'")
    if backbone["cache_dtype"] != "float16":
        raise ConfigError("'backbone.cache_dtype' must be 'float16' for MVP")
    _require_bool(backbone, "frozen")
    if not backbone["frozen"]:
        raise ConfigError("'backbone.frozen' must be true for MVP")
    _require_int(backbone, "cache_batch_size")
    _require_int(backbone, "cache_max_images_per_split", minimum=0)

    model = _require_mapping(config, "model")
    hidden_dim = _require_int(model, "hidden_dim")
    heads = _require_int(model, "attention_heads")
    if hidden_dim % heads != 0:
        raise ConfigError(
            f"'model.hidden_dim' ({hidden_dim}) must be divisible by "
            f"'model.attention_heads' ({heads})"
        )
    _require_int(model, "layers")
    _require_real(model, "mlp_ratio", minimum=1.0)
    _require_real(model, "dropout", minimum=0.0, maximum=1.0)
    _require_bool(model, "use_frame_index_embedding")
    _require_bool(model, "use_native_sdpa")

    training = _require_mapping(config, "training")
    batch_size = _require_int(training, "batch_size")
    _require_int(training, "gradient_accumulation")
    min_steps = _require_int(training, "min_steps")
    max_steps = _require_int(training, "max_steps")
    if min_steps > max_steps:
        raise ConfigError("'training.min_steps' cannot exceed 'training.max_steps'")
    for key in (
        "validate_every_steps",
        "checkpoint_every_steps",
        "early_stop_patience",
    ):
        _require_int(training, key)
    warmup_steps = _require_int(training, "warmup_steps", minimum=0)
    if warmup_steps >= max_steps:
        raise ConfigError("'training.warmup_steps' must be less than max_steps")
    _require_real(training, "learning_rate", minimum=0.0)
    _require_real(training, "weight_decay", minimum=0.0)
    _require_real(training, "max_grad_norm", minimum=0.0)
    _require_real(
        training,
        "label_smoothing",
        minimum=0.0,
        maximum=1.0,
    )
    _require_real(
        training,
        "maskgit_min_mask_ratio",
        minimum=0.0,
        maximum=1.0,
    )
    if training.get("precision") not in {"bf16", "fp16", "fp32"}:
        raise ConfigError("'training.precision' must be bf16, fp16, or fp32")

    decoding = _require_mapping(config, "decoding")
    default_steps = _require_int(decoding, "default_steps")
    evaluation_steps = decoding.get("evaluation_steps")
    if not isinstance(evaluation_steps, list) or not evaluation_steps:
        raise ConfigError("'decoding.evaluation_steps' must be a non-empty list")
    normalized_steps: list[int] = []
    for value in evaluation_steps:
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ConfigError(
                "'decoding.evaluation_steps' must contain positive integers"
            )
        normalized_steps.append(int(value))
    if default_steps not in normalized_steps:
        raise ConfigError(
            "'decoding.default_steps' must appear in evaluation_steps"
        )
    if decoding.get("schedule") != "cosine":
        raise ConfigError("'decoding.schedule' must be 'cosine' for MVP")
    evidence = _require_mapping(decoding, "evidence")
    top_k_frames = _require_int(evidence, "top_k_frames")
    if top_k_frames > condition_frames:
        raise ConfigError(
            "'decoding.evidence.top_k_frames' cannot exceed condition_frames"
        )
    lambda_grid = evidence.get("lambda_grid")
    if (
        not isinstance(lambda_grid, list)
        or not lambda_grid
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or value < 0
            for value in lambda_grid
        )
    ):
        raise ConfigError(
            "'decoding.evidence.lambda_grid' must contain non-negative numbers"
        )
    _require_bool(evidence, "stop_gradient")
    _require_bool(evidence, "enable_remask")
    default_lambda = _require_real(
        evidence,
        "default_lambda",
        minimum=0.0,
    )
    if default_lambda not in [float(value) for value in lambda_grid]:
        raise ConfigError(
            "'decoding.evidence.default_lambda' must appear in lambda_grid"
        )

    evaluation = _require_mapping(config, "evaluation")
    _require_int(evaluation, "batch_size")
    far_targets = evaluation.get("far_targets")
    if (
        not isinstance(far_targets, list)
        or not far_targets
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not 0.0 < float(value) < 1.0
            for value in far_targets
        )
    ):
        raise ConfigError("'evaluation.far_targets' must contain values in (0, 1)")
    scenarios = evaluation.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ConfigError("'evaluation.scenarios' must be a non-empty list")
    required_scenarios = {
        "clean",
        "low_quality",
        "complementary_occlusion",
        "common_occlusion",
        "wrong_identity",
    }
    missing_scenarios = required_scenarios.difference(scenarios)
    if missing_scenarios:
        raise ConfigError(
            "'evaluation.scenarios' is missing required MVP scenarios: "
            + ", ".join(sorted(missing_scenarios))
        )
    for key in (
        "latency_warmup_runs",
        "latency_measurement_runs",
        "permutation_trials",
        "bootstrap_samples",
    ):
        _require_int(evaluation, key)

    runtime = _require_mapping(config, "runtime")
    _require_bool(runtime, "require_cuda")
    _require_bool(runtime, "deterministic")
    _require_int(runtime, "num_workers", minimum=0)
    _require_bool(runtime, "pin_memory")
    max_peak_memory_gib = _require_real(
        runtime,
        "max_peak_memory_gib",
        minimum=0.001,
        maximum=7.2,
    )
    if max_peak_memory_gib <= 0:
        raise ConfigError("'runtime.max_peak_memory_gib' must be positive")
    _validate_positive_int_sequence(
        runtime.get("oom_batch_fallback"),
        name="runtime.oom_batch_fallback",
        expected_first=batch_size,
    )
    _validate_positive_int_sequence(
        runtime.get("oom_condition_frames_fallback"),
        name="runtime.oom_condition_frames_fallback",
        expected_first=condition_frames,
    )
    _validate_positive_int_sequence(
        runtime.get("oom_hidden_dim_fallback"),
        name="runtime.oom_hidden_dim_fallback",
        expected_first=hidden_dim,
    )

    logging = _require_mapping(config, "logging")
    for key in ("jsonl_filename", "metrics_filename", "manifest_filename"):
        value = logging.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"'logging.{key}' must be a non-empty filename")
        if Path(value).name != value:
            raise ConfigError(f"'logging.{key}' must be a filename, not a path")


def _resolve_workspace_paths(
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    for key, raw_value in resolved["paths"].items():
        path = Path(raw_value).expanduser()
        if not path.is_absolute():
            path = project_root / path
        resolved["paths"][key] = str(path.resolve())
    return resolved


def load_config(config_path: str | Path, profile: str) -> ResolvedConfig:
    """Load the YAML file, deep-merge one profile, resolve paths, and validate."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, Mapping):
        raise ConfigError("configuration root must be a mapping")
    if document.get("version") != 1:
        raise ConfigError("configuration 'version' must be 1")

    defaults = document.get("defaults")
    profiles = document.get("profiles")
    if not isinstance(defaults, Mapping):
        raise ConfigError("'defaults' must be a mapping")
    if not isinstance(profiles, Mapping):
        raise ConfigError("'profiles' must be a mapping")
    if profile not in profiles:
        available = ", ".join(sorted(str(name) for name in profiles))
        raise ConfigError(
            f"unknown profile '{profile}'; available profiles: {available}"
        )
    override = profiles[profile]
    if not isinstance(override, Mapping):
        raise ConfigError(f"profile '{profile}' must be a mapping")

    project_root = path.parent.parent.resolve()
    merged = _deep_merge(defaults, override)
    merged.setdefault("runtime", {})["profile"] = profile
    merged = _resolve_workspace_paths(merged, project_root)
    validate_config(merged)
    return ResolvedConfig(
        values=merged,
        profile=profile,
        config_path=path,
        project_root=project_root,
    )
