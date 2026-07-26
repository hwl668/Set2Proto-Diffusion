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
        "p2_2",
        "p2_3",
        "p2_4",
        "p2_5",
        "p2_6",
        "p3_0",
        "p3_1",
        "pointer_route",
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

    p2_2 = _require_mapping(config, "p2_2")
    if p2_2.get("protocol") != "route_identity_carveout_rollout_stability":
        raise ConfigError(
            "'p2_2.protocol' must be "
            "'route_identity_carveout_rollout_stability'"
        )
    expected_splits = {
        "source_split": "train",
        "codebook_fit_split": "route_train",
        "calibration_split": "route_calibration",
        "validation_split": "route_validation",
    }
    for key, expected in expected_splits.items():
        if p2_2.get(key) != expected:
            raise ConfigError(f"'p2_2.{key}' must be '{expected}'")
    if p2_2.get("construct_test_artifacts") is not False:
        raise ConfigError("'p2_2.construct_test_artifacts' must be false")
    if p2_2.get("anchor") != "quality_pooling":
        raise ConfigError("'p2_2.anchor' must be 'quality_pooling'")
    if p2_2.get("residual_definition") != "raw_teacher_minus_anchor":
        raise ConfigError(
            "'p2_2.residual_definition' must be "
            "'raw_teacher_minus_anchor'"
        )
    for key in (
        "codebook_size",
        "max_fit_tokens",
        "codebook_batch_size",
        "codebook_iterations",
        "codebook_n_init",
        "assignment_chunk_size",
        "smoke_steps",
        "formal_steps",
        "checkpoint_every_steps",
        "batch_size",
        "gradient_accumulation",
    ):
        _require_int(p2_2, key)
    if int(p2_2["smoke_steps"]) >= int(p2_2["formal_steps"]):
        raise ConfigError("'p2_2.smoke_steps' must be less than formal_steps")
    if p2_2.get("fit_device") not in {"auto", "cpu", "cuda"}:
        raise ConfigError("'p2_2.fit_device' must be auto, cpu, or cuda")
    checkpoint_steps = p2_2.get("calibration_checkpoint_steps")
    if (
        not isinstance(checkpoint_steps, list)
        or not checkpoint_steps
        or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) <= 0
            or int(value) > int(p2_2["formal_steps"])
            for value in checkpoint_steps
        )
    ):
        raise ConfigError(
            "'p2_2.calibration_checkpoint_steps' contains invalid steps"
        )
    if len({int(value) for value in checkpoint_steps}) != len(
        checkpoint_steps
    ):
        raise ConfigError(
            "'p2_2.calibration_checkpoint_steps' must be unique"
        )
    if int(p2_2["formal_steps"]) not in {
        int(value) for value in checkpoint_steps
    }:
        raise ConfigError(
            "'p2_2.calibration_checkpoint_steps' must include formal_steps"
        )
    if any(
        int(value) % int(p2_2["checkpoint_every_steps"]) != 0
        for value in checkpoint_steps
    ):
        raise ConfigError(
            "P2-2 calibration checkpoints must align with checkpoint cadence"
        )
    for key in ("learning_rate", "max_grad_norm"):
        _require_real(p2_2, key, minimum=1e-12)
    _require_real(p2_2, "weight_decay", minimum=0.0)
    warmup = _require_int(p2_2, "warmup_steps", minimum=0)
    if warmup >= int(p2_2["formal_steps"]):
        raise ConfigError("'p2_2.warmup_steps' must be less than formal_steps")
    if p2_2.get("precision") not in {"bf16", "fp16"}:
        raise ConfigError("'p2_2.precision' must be bf16 or fp16")
    _require_bool(p2_2, "use_gpu_resident_cache")
    if p2_2["use_gpu_resident_cache"] is not True:
        raise ConfigError("P2-2 requires a GPU-resident training cache")
    p2_2_corruption = _require_mapping(p2_2, "corruption")
    probabilities = [
        _require_real(
            p2_2_corruption,
            key,
            minimum=0.0,
            maximum=1.0,
        )
        for key in (
            "all_mask_probability",
            "teacher_partial_probability",
            "self_rollout_probability",
        )
    ]
    if abs(sum(probabilities) - 1.0) > 1e-8:
        raise ConfigError("P2-2 corruption probabilities must sum to 1")
    _require_real(
        p2_2_corruption,
        "min_mask_ratio",
        minimum=1e-8,
        maximum=1.0,
    )
    rollout_total_steps = _require_int(
        p2_2_corruption,
        "rollout_total_steps",
        minimum=2,
    )
    rollout_depths = p2_2_corruption.get("rollout_depths")
    if (
        not isinstance(rollout_depths, list)
        or not rollout_depths
        or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 1 <= int(value) < rollout_total_steps
            for value in rollout_depths
        )
    ):
        raise ConfigError("'p2_2.corruption.rollout_depths' is invalid")
    if len({int(value) for value in rollout_depths}) != len(rollout_depths):
        raise ConfigError(
            "'p2_2.corruption.rollout_depths' must be unique"
        )
    if p2_2_corruption.get("rollout_depth_sampling") != "uniform":
        raise ConfigError(
            "'p2_2.corruption.rollout_depth_sampling' must be 'uniform'"
        )
    _require_bool(p2_2_corruption, "stop_gradient")
    if p2_2_corruption["stop_gradient"] is not True:
        raise ConfigError("P2-2 self-rollout must use stop-gradient")
    for key in (
        "masked_token_weight",
        "correct_committed_weight",
        "wrong_committed_weight",
    ):
        _require_real(p2_2_corruption, key, minimum=0.0)
    if float(p2_2_corruption["masked_token_weight"]) <= 0.0:
        raise ConfigError("P2-2 masked_token_weight must be positive")
    if (
        float(p2_2_corruption["wrong_committed_weight"])
        < float(p2_2_corruption["correct_committed_weight"])
    ):
        raise ConfigError(
            "P2-2 wrong committed weight cannot be below stability weight"
        )
    p2_2_evidence = _require_mapping(p2_2, "evidence")
    lambdas = p2_2_evidence.get("lambda_candidates")
    if (
        not isinstance(lambdas, list)
        or not lambdas
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or float(value) < 0.0
            for value in lambdas
        )
    ):
        raise ConfigError(
            "'p2_2.evidence.lambda_candidates' must be non-negative"
        )
    if len({float(value) for value in lambdas}) != len(lambdas):
        raise ConfigError(
            "'p2_2.evidence.lambda_candidates' must be unique"
        )
    _require_real(
        p2_2_evidence,
        "auc_tie_tolerance",
        minimum=0.0,
    )
    p2_2_top_k = _require_int(p2_2_evidence, "top_k_frames")
    if p2_2_top_k > condition_frames:
        raise ConfigError(
            "'p2_2.evidence.top_k_frames' cannot exceed condition_frames"
        )
    _require_real(
        p2_2_evidence,
        "identity_neighbor_fraction",
        minimum=1e-8,
        maximum=1.0,
    )
    for key in (
        "identity_temperature",
        "quality_weight",
        "local_temperature",
    ):
        _require_real(p2_2_evidence, key, minimum=0.0)
    _require_real(
        p2_2_evidence,
        "reliability_floor",
        minimum=1e-12,
    )
    if p2_2_evidence.get("primary_mode") != "two_level_remask":
        raise ConfigError(
            "'p2_2.evidence.primary_mode' must be 'two_level_remask'"
        )
    p2_2_gates = _require_mapping(p2_2, "gates")
    for key in (
        "min_residual_oracle_hard_auc_gain_over_quality",
        "min_four_step_gain_over_one_shot",
        "min_guided_gain_over_confidence_four_step",
        "min_guided_gain_over_one_shot",
        "max_clean_auc_drop_vs_confidence",
        "max_permutation_difference",
    ):
        _require_real(p2_2_gates, key, minimum=0.0)
    _require_int(p2_2_gates, "min_hard_scenario_wins")
    _require_bool(p2_2_gates, "require_positive_step_net_correction")

    p2_3 = _require_mapping(config, "p2_3")
    if (
        p2_3.get("protocol")
        != "route_identity_carveout_risk_controlled_refinement"
    ):
        raise ConfigError(
            "'p2_3.protocol' must be "
            "'route_identity_carveout_risk_controlled_refinement'"
        )
    for key, expected in (
        ("source_split", "train"),
        ("gate_train_split", "route_train"),
        ("calibration_split", "route_calibration"),
        ("validation_split", "route_validation"),
        ("base_checkpoint_source", "p2_2_selected_rollout"),
    ):
        if p2_3.get(key) != expected:
            raise ConfigError(f"'p2_3.{key}' must be '{expected}'")
    if p2_3.get("construct_test_artifacts") is not False:
        raise ConfigError("'p2_3.construct_test_artifacts' must be false")
    _require_bool(p2_3, "frozen_transformer")
    if p2_3["frozen_transformer"] is not True:
        raise ConfigError("P2-3 requires a frozen Transformer")
    p2_3_gate = _require_mapping(p2_3, "gate")
    for key in (
        "hidden_dim",
        "batch_size",
        "smoke_steps",
        "formal_steps",
        "feature_batch_size",
        "views_per_fraction",
    ):
        _require_int(p2_3_gate, key)
    if int(p2_3_gate["smoke_steps"]) >= int(p2_3_gate["formal_steps"]):
        raise ConfigError("'p2_3.gate.smoke_steps' must be less than formal_steps")
    for key in ("learning_rate", "max_grad_norm"):
        _require_real(p2_3_gate, key, minimum=1e-12)
    _require_real(p2_3_gate, "weight_decay", minimum=0.0)
    _require_real(p2_3_gate, "dropout", minimum=0.0, maximum=1.0)
    if float(p2_3_gate["dropout"]) >= 1.0:
        raise ConfigError("'p2_3.gate.dropout' must be less than 1")
    training_mask_fractions = p2_3_gate.get("training_mask_fractions")
    if (
        not isinstance(training_mask_fractions, list)
        or not training_mask_fractions
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not 0.0 < float(value) <= 1.0
            for value in training_mask_fractions
        )
    ):
        raise ConfigError(
            "'p2_3.gate.training_mask_fractions' contains invalid values"
        )
    if len({float(value) for value in training_mask_fractions}) != len(
        training_mask_fractions
    ):
        raise ConfigError(
            "'p2_3.gate.training_mask_fractions' must be unique"
        )
    _require_real(
        p2_3_gate,
        "random_view_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    _require_real(
        p2_3_gate,
        "proposal_evidence_weight",
        minimum=0.0,
    )
    p2_3_calibration = _require_mapping(p2_3, "calibration")
    for key in (
        "proposal_fraction_candidates",
        "utility_threshold_candidates",
    ):
        values = p2_3_calibration.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or float(value) < 0.0
                or (
                    key == "proposal_fraction_candidates"
                    and float(value) <= 0.0
                )
                or float(value) > 1.0
                for value in values
            )
        ):
            raise ConfigError(f"'p2_3.calibration.{key}' is invalid")
        if len({float(value) for value in values}) != len(values):
            raise ConfigError(f"'p2_3.calibration.{key}' must be unique")
    round_candidates = p2_3_calibration.get("round_candidates")
    if (
        not isinstance(round_candidates, list)
        or not round_candidates
        or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) <= 0
            for value in round_candidates
        )
    ):
        raise ConfigError("'p2_3.calibration.round_candidates' is invalid")
    if len({int(value) for value in round_candidates}) != len(round_candidates):
        raise ConfigError(
            "'p2_3.calibration.round_candidates' must be unique"
        )
    _require_real(
        p2_3_calibration,
        "auc_tie_tolerance",
        minimum=0.0,
    )
    p2_3_gates = _require_mapping(p2_3, "gates")
    for key in (
        "min_hard_auc_gain_over_rollout_one_shot",
        "min_hard_auc_gain_over_matched_one_shot",
        "max_clean_auc_drop",
        "max_permutation_difference",
    ):
        _require_real(p2_3_gates, key, minimum=0.0)
    _require_int(p2_3_gates, "min_hard_scenario_wins")
    _require_bool(p2_3_gates, "require_positive_net_correction")

    p2_4 = _require_mapping(config, "p2_4")
    if (
        p2_4.get("protocol")
        != "route_identity_carveout_residual_proposal_oracle"
    ):
        raise ConfigError(
            "'p2_4.protocol' must be "
            "'route_identity_carveout_residual_proposal_oracle'"
        )
    for key, expected in (
        ("source_split", "train"),
        ("calibration_split", "route_calibration"),
        ("validation_split", "route_validation"),
        ("base_checkpoint_source", "p2_2_selected_rollout"),
    ):
        if p2_4.get(key) != expected:
            raise ConfigError(f"'p2_4.{key}' must be '{expected}'")
    if p2_4.get("construct_test_artifacts") is not False:
        raise ConfigError("'p2_4.construct_test_artifacts' must be false")
    _require_bool(p2_4, "frozen_transformer")
    if p2_4["frozen_transformer"] is not True:
        raise ConfigError("P2-4 requires a frozen Transformer")
    _require_real(p2_4, "proposal_evidence_weight", minimum=0.0)
    proposal_fractions = p2_4.get("proposal_fraction_candidates")
    if (
        not isinstance(proposal_fractions, list)
        or not proposal_fractions
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not 0.0 < float(value) <= 1.0
            for value in proposal_fractions
        )
    ):
        raise ConfigError(
            "'p2_4.proposal_fraction_candidates' is invalid"
        )
    if len({float(value) for value in proposal_fractions}) != len(
        proposal_fractions
    ):
        raise ConfigError(
            "'p2_4.proposal_fraction_candidates' must be unique"
        )
    top_k_candidates = p2_4.get("top_k_candidates")
    if (
        not isinstance(top_k_candidates, list)
        or not top_k_candidates
        or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) <= 0
            or int(value) > int(config["quantization"]["codebook_size"])
            for value in top_k_candidates
        )
    ):
        raise ConfigError("'p2_4.top_k_candidates' is invalid")
    if len({int(value) for value in top_k_candidates}) != len(
        top_k_candidates
    ):
        raise ConfigError("'p2_4.top_k_candidates' must be unique")
    p2_4_rounds = p2_4.get("round_candidates")
    if (
        not isinstance(p2_4_rounds, list)
        or not p2_4_rounds
        or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) <= 0
            for value in p2_4_rounds
        )
    ):
        raise ConfigError("'p2_4.round_candidates' is invalid")
    if len({int(value) for value in p2_4_rounds}) != len(p2_4_rounds):
        raise ConfigError("'p2_4.round_candidates' must be unique")
    _require_real(p2_4, "auc_tie_tolerance", minimum=0.0)
    p2_4_gates = _require_mapping(p2_4, "gates")
    for key in (
        "min_exact_oracle_hard_auc_gain_over_matched_one_shot",
        "min_local_oracle_hard_auc_gain_over_matched_one_shot",
        "min_wrong_token_correctable_fraction",
        "max_permutation_difference",
    ):
        _require_real(p2_4_gates, key, minimum=0.0)
    _require_int(p2_4_gates, "min_hard_scenario_wins")

    p2_5 = _require_mapping(config, "p2_5")
    if (
        p2_5.get("protocol")
        != "route_identity_carveout_listwise_identity_reranker"
    ):
        raise ConfigError(
            "'p2_5.protocol' must be "
            "'route_identity_carveout_listwise_identity_reranker'"
        )
    for key, expected in (
        ("source_split", "train"),
        ("reranker_train_split", "route_train"),
        ("calibration_split", "route_calibration"),
        ("validation_split", "route_validation"),
        ("base_checkpoint_source", "p2_2_selected_rollout"),
    ):
        if p2_5.get(key) != expected:
            raise ConfigError(f"'p2_5.{key}' must be '{expected}'")
    if p2_5.get("construct_test_artifacts") is not False:
        raise ConfigError("'p2_5.construct_test_artifacts' must be false")
    _require_bool(p2_5, "frozen_transformer")
    if p2_5["frozen_transformer"] is not True:
        raise ConfigError("P2-5 requires a frozen Transformer")
    p2_5_top_k = _require_int(p2_5, "top_k")
    if p2_5_top_k > int(config["quantization"]["codebook_size"]):
        raise ConfigError("'p2_5.top_k' cannot exceed codebook size")
    _require_int(p2_5, "feature_batch_size")
    _require_int(p2_5, "hard_impostors")
    p2_5_utility = _require_mapping(p2_5, "utility")
    for key in (
        "local_teacher_weight",
        "gallery_margin_weight",
        "exact_token_weight",
    ):
        _require_real(p2_5_utility, key, minimum=0.0)
    _require_real(p2_5_utility, "temperature", minimum=1e-12)
    if (
        float(p2_5_utility["local_teacher_weight"])
        + float(p2_5_utility["gallery_margin_weight"])
        + float(p2_5_utility["exact_token_weight"])
        <= 0.0
    ):
        raise ConfigError("P2-5 utility must contain positive supervision")
    p2_5_reranker = _require_mapping(p2_5, "reranker")
    for key in (
        "hidden_dim",
        "batch_size",
        "smoke_steps",
        "formal_steps",
    ):
        _require_int(p2_5_reranker, key)
    if int(p2_5_reranker["smoke_steps"]) >= int(
        p2_5_reranker["formal_steps"]
    ):
        raise ConfigError(
            "'p2_5.reranker.smoke_steps' must be less than formal_steps"
        )
    for key in (
        "learning_rate",
        "max_grad_norm",
        "listwise_loss_weight",
        "identity_margin_loss_weight",
        "local_map_loss_weight",
    ):
        _require_real(p2_5_reranker, key, minimum=1e-12)
    _require_real(p2_5_reranker, "weight_decay", minimum=0.0)
    _require_real(
        p2_5_reranker,
        "dropout",
        minimum=0.0,
        maximum=1.0,
    )
    if float(p2_5_reranker["dropout"]) >= 1.0:
        raise ConfigError("'p2_5.reranker.dropout' must be less than 1")
    _require_real(p2_5_reranker, "identity_margin", minimum=0.0)
    p2_5_calibration = _require_mapping(p2_5, "calibration")
    for key in (
        "model_logit_weight_candidates",
        "replacement_threshold_candidates",
    ):
        values = p2_5_calibration.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or float(value) < 0.0
                for value in values
            )
        ):
            raise ConfigError(f"'p2_5.calibration.{key}' is invalid")
        if len({float(value) for value in values}) != len(values):
            raise ConfigError(f"'p2_5.calibration.{key}' must be unique")
    _require_real(
        p2_5_calibration,
        "auc_tie_tolerance",
        minimum=0.0,
    )
    p2_5_gates = _require_mapping(p2_5, "gates")
    for key in (
        "min_hard_auc_gain_over_rollout_one_shot",
        "min_hard_auc_gain_over_matched_one_shot",
        "max_clean_auc_drop",
        "max_permutation_difference",
    ):
        _require_real(p2_5_gates, key, minimum=0.0)
    _require_int(p2_5_gates, "min_hard_scenario_wins")
    _require_bool(p2_5_gates, "require_positive_net_correction")

    p2_6 = _require_mapping(config, "p2_6")
    if (
        p2_6.get("protocol")
        != "route_identity_carveout_set_aware_global_reranker"
    ):
        raise ConfigError(
            "'p2_6.protocol' must be "
            "'route_identity_carveout_set_aware_global_reranker'"
        )
    for key, expected in (
        ("source_split", "train"),
        ("reranker_train_split", "route_train"),
        ("calibration_split", "route_calibration"),
        ("validation_split", "route_validation"),
        ("frozen_candidate_cache", "p2_5_route_train_top8"),
        ("base_checkpoint_source", "p2_2_selected_rollout"),
    ):
        if p2_6.get(key) != expected:
            raise ConfigError(f"'p2_6.{key}' must be '{expected}'")
    if p2_6.get("construct_test_artifacts") is not False:
        raise ConfigError("'p2_6.construct_test_artifacts' must be false")
    _require_bool(p2_6, "frozen_transformer")
    if p2_6["frozen_transformer"] is not True:
        raise ConfigError("P2-6 requires a frozen Transformer")
    p2_6_top_k = _require_int(p2_6, "top_k")
    if p2_6_top_k != int(p2_5["top_k"]):
        raise ConfigError("'p2_6.top_k' must match the frozen P2-5 cache")
    p2_6_reranker = _require_mapping(p2_6, "reranker")
    for key in (
        "hidden_dim",
        "attention_heads",
        "layers",
        "feedforward_multiplier",
        "batch_size",
        "smoke_steps",
        "formal_steps",
    ):
        _require_int(p2_6_reranker, key)
    if int(p2_6_reranker["hidden_dim"]) % int(
        p2_6_reranker["attention_heads"]
    ):
        raise ConfigError(
            "'p2_6.reranker.hidden_dim' must be divisible by attention_heads"
        )
    if int(p2_6_reranker["smoke_steps"]) >= int(
        p2_6_reranker["formal_steps"]
    ):
        raise ConfigError(
            "'p2_6.reranker.smoke_steps' must be less than formal_steps"
        )
    for key in (
        "learning_rate",
        "max_grad_norm",
        "listwise_loss_weight",
        "identity_margin_loss_weight",
        "local_map_loss_weight",
        "replacement_gate_loss_weight",
        "expected_injury_loss_weight",
    ):
        _require_real(p2_6_reranker, key, minimum=1e-12)
    _require_real(p2_6_reranker, "weight_decay", minimum=0.0)
    _require_real(
        p2_6_reranker,
        "dropout",
        minimum=0.0,
        maximum=1.0,
    )
    if float(p2_6_reranker["dropout"]) >= 1.0:
        raise ConfigError("'p2_6.reranker.dropout' must be less than 1")
    _require_real(p2_6_reranker, "identity_margin", minimum=0.0)
    _require_real(
        p2_6_reranker,
        "safe_replacement_probability_gain",
        minimum=0.0,
        maximum=1.0,
    )
    p2_6_calibration = _require_mapping(p2_6, "calibration")
    for key in (
        "model_logit_weight_candidates",
        "replacement_threshold_candidates",
        "gate_threshold_candidates",
        "replacement_budget_fraction_candidates",
    ):
        values = p2_6_calibration.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or float(value) < 0.0
                for value in values
            )
        ):
            raise ConfigError(f"'p2_6.calibration.{key}' is invalid")
        if len({float(value) for value in values}) != len(values):
            raise ConfigError(f"'p2_6.calibration.{key}' must be unique")
    for key in (
        "gate_threshold_candidates",
        "replacement_budget_fraction_candidates",
    ):
        if any(
            float(value) > 1.0
            for value in p2_6_calibration[key]
        ):
            raise ConfigError(
                f"'p2_6.calibration.{key}' cannot exceed 1"
            )
    _require_real(p2_6_calibration, "auc_tie_tolerance", minimum=0.0)
    p2_6_gates = _require_mapping(p2_6, "gates")
    for key in (
        "min_hard_auc_gain_over_rollout_one_shot",
        "min_hard_auc_gain_over_matched_one_shot",
        "max_clean_auc_drop",
        "max_injured_fraction",
        "max_permutation_difference",
    ):
        _require_real(p2_6_gates, key, minimum=0.0)
    _require_int(p2_6_gates, "min_hard_scenario_wins")
    _require_bool(p2_6_gates, "require_positive_net_correction")

    p3_0 = _require_mapping(config, "p3_0")
    if (
        p3_0.get("protocol")
        != "frozen_evidence_anchor_residual_quantization"
    ):
        raise ConfigError(
            "'p3_0.protocol' must be "
            "'frozen_evidence_anchor_residual_quantization'"
        )
    for key, expected in (
        ("train_split", "train"),
        ("selection_split", "val"),
        ("evaluation_split", "test"),
        (
            "frozen_scalar_router",
            "stage16_p1_3_scalar_frozen_step550",
        ),
        ("residual_definition", "raw_teacher_minus_anchor"),
        ("matched_quality_codebook", "p2_1_expanded_residual"),
    ):
        if p3_0.get(key) != expected:
            raise ConfigError(f"'p3_0.{key}' must be '{expected}'")
    _require_bool(p3_0, "construct_test_after_validation_lock")
    if p3_0["construct_test_after_validation_lock"] is not True:
        raise ConfigError("P3-0 requires lock-before-test evaluation")
    for key in (
        "codebook_size",
        "max_fit_tokens",
        "batch_size",
        "iterations",
        "n_init",
        "assignment_chunk_size",
        "inference_batch_size",
    ):
        _require_int(p3_0, key)
    if int(p3_0["codebook_size"]) != 1024:
        raise ConfigError("'p3_0.codebook_size' must remain fixed at 1024")
    _require_int(p3_0, "seed", minimum=0)
    if p3_0.get("fit_device") not in {"auto", "cuda", "cpu"}:
        raise ConfigError("'p3_0.fit_device' must be auto, cuda, or cpu")
    p3_0_gates = _require_mapping(p3_0, "gates")
    for key in (
        "min_train_residual_norm_reduction",
        "min_val_anchor_hard_auc_gain",
        "min_test_anchor_hard_auc_gain",
        "min_val_quantized_map_cosine_gain",
        "min_test_quantized_map_cosine_gain",
        "max_val_quantized_hard_auc_drop",
        "max_test_quantized_hard_auc_drop",
        "max_permutation_difference",
    ):
        _require_real(p3_0_gates, key, minimum=0.0)
    _require_int(p3_0_gates, "min_scenario_map_cosine_wins")

    p3_1 = _require_mapping(config, "p3_1")
    if p3_1.get("protocol") != "evidence_anchor_parameter_matched_maskgit":
        raise ConfigError(
            "'p3_1.protocol' must be "
            "'evidence_anchor_parameter_matched_maskgit'"
        )
    for key, expected in (
        ("train_split", "train"),
        ("selection_split", "val"),
        ("evaluation_split", "test"),
        (
            "frozen_scalar_router",
            "stage16_p1_3_scalar_frozen_step550",
        ),
        ("frozen_codebook", "p2_1_quality_residual_k1024"),
        ("residual_definition", "raw_teacher_minus_evidence_anchor"),
    ):
        if p3_1.get(key) != expected:
            raise ConfigError(f"'p3_1.{key}' must be '{expected}'")
    _require_bool(p3_1, "construct_test_after_validation_lock")
    if p3_1["construct_test_after_validation_lock"] is not True:
        raise ConfigError("P3-1 requires lock-before-test evaluation")
    if _require_int(p3_1, "codebook_size") != 1024:
        raise ConfigError("'p3_1.codebook_size' must remain fixed at 1024")
    p3_1_model = _require_mapping(p3_1, "model")
    for key in ("hidden_dim", "layers", "attention_heads"):
        _require_int(p3_1_model, key)
    if int(p3_1_model["hidden_dim"]) % int(
        p3_1_model["attention_heads"]
    ):
        raise ConfigError("P3-1 hidden_dim must be divisible by attention_heads")
    _require_real(p3_1_model, "mlp_ratio", minimum=1e-12)
    _require_real(p3_1_model, "dropout", minimum=0.0, maximum=1.0)
    p3_1_training = _require_mapping(p3_1, "training")
    for key in (
        "batch_size",
        "gradient_accumulation",
        "smoke_steps",
        "formal_steps",
        "checkpoint_every_steps",
        "warmup_steps",
    ):
        _require_int(p3_1_training, key)
    if int(p3_1_training["smoke_steps"]) >= int(
        p3_1_training["formal_steps"]
    ):
        raise ConfigError("P3-1 smoke_steps must be less than formal_steps")
    for key in (
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "label_smoothing",
        "maskgit_min_mask_ratio",
    ):
        _require_real(p3_1_training, key, minimum=0.0)
    if p3_1_training.get("precision") not in {"bf16", "fp16", "fp32"}:
        raise ConfigError("'p3_1.training.precision' is invalid")
    p3_1_decoding = _require_mapping(p3_1, "decoding")
    steps = p3_1_decoding.get("step_candidates")
    if (
        not isinstance(steps, list)
        or not steps
        or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 1
            for value in steps
        )
    ):
        raise ConfigError("'p3_1.decoding.step_candidates' is invalid")
    if 4 not in {int(value) for value in steps}:
        raise ConfigError("P3-1 step candidates must include four steps")
    modes = p3_1_decoding.get("evidence_modes")
    if (
        not isinstance(modes, list)
        or set(modes) != {"evidence-ordering", "evidence-logits"}
    ):
        raise ConfigError("'p3_1.decoding.evidence_modes' is invalid")
    lambdas = p3_1_decoding.get("evidence_lambda_candidates")
    if (
        not isinstance(lambdas, list)
        or not lambdas
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or float(value) < 0.0
            for value in lambdas
        )
    ):
        raise ConfigError(
            "'p3_1.decoding.evidence_lambda_candidates' is invalid"
        )
    _require_int(p3_1_decoding, "top_k_frames")
    _require_real(p3_1_decoding, "auc_tie_tolerance", minimum=0.0)
    p3_1_gates = _require_mapping(p3_1, "gates")
    for key in (
        "min_one_shot_hard_auc_gain_over_anchor",
        "min_maskgit_gain_over_one_shot",
        "min_evidence_gain_over_confidence",
        "max_clean_auc_drop",
        "max_permutation_difference",
    ):
        _require_real(p3_1_gates, key, minimum=0.0)
    _require_int(p3_1_gates, "min_hard_scenario_wins_over_one_shot")

    pointer_route = _require_mapping(config, "pointer_route")
    if pointer_route.get("protocol") != "train_identity_carveout":
        raise ConfigError(
            "'pointer_route.protocol' must be 'train_identity_carveout'"
        )
    if pointer_route.get("source_split") != "train":
        raise ConfigError("'pointer_route.source_split' must be 'train'")
    if pointer_route.get("anchor") != "quality_pooling":
        raise ConfigError("'pointer_route.anchor' must be 'quality_pooling'")
    route_counts = [
        _require_int(pointer_route, "route_train_identities"),
        _require_int(pointer_route, "route_calibration_identities"),
        _require_int(pointer_route, "route_validation_identities"),
    ]
    if sum(route_counts) != int(real["train_identities"]):
        raise ConfigError(
            "pointer-route identity counts must exactly partition "
            "'data.real.train_identities'"
        )
    _require_int(pointer_route, "seed_offset", minimum=0)
    for key in ("delta_candidates", "temperature_candidates"):
        values = pointer_route.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or float(value) <= 0.0
                for value in values
            )
        ):
            raise ConfigError(
                f"'pointer_route.{key}' must contain positive numbers"
            )
        if len({float(value) for value in values}) != len(values):
            raise ConfigError(
                f"'pointer_route.{key}' must not contain duplicates"
            )
    _require_real(
        pointer_route,
        "auc_tie_tolerance",
        minimum=0.0,
        maximum=1.0,
    )
    _require_real(
        pointer_route,
        "equivalent_epsilon",
        minimum=0.0,
        maximum=2.0,
    )
    pointer_route_gates = _require_mapping(pointer_route, "gates")
    _require_real(
        pointer_route_gates,
        "min_oracle_hard_auc_gain",
        minimum=0.0,
        maximum=1.0,
    )
    _require_int(pointer_route_gates, "min_hard_scenario_wins")
    for key in (
        "min_non_anchor_fraction",
        "min_teacher_loo_equivalent_agreement",
        "max_wrong_frame_selection_fraction",
    ):
        _require_real(
            pointer_route_gates,
            key,
            minimum=0.0,
            maximum=1.0,
        )
    pointer_one_shot = _require_mapping(pointer_route, "one_shot")
    for key in (
        "batch_size",
        "gradient_accumulation",
        "smoke_steps",
        "min_steps",
        "max_steps",
        "validate_every_steps",
        "early_stop_patience",
        "warmup_steps",
        "max_parameters",
    ):
        minimum = 0 if key == "warmup_steps" else 1
        _require_int(pointer_one_shot, key, minimum=minimum)
    if int(pointer_one_shot["smoke_steps"]) >= int(
        pointer_one_shot["min_steps"]
    ):
        raise ConfigError(
            "'pointer_route.one_shot.smoke_steps' must be less than min_steps"
        )
    if int(pointer_one_shot["min_steps"]) > int(
        pointer_one_shot["max_steps"]
    ):
        raise ConfigError(
            "'pointer_route.one_shot.min_steps' cannot exceed max_steps"
        )
    if int(pointer_one_shot["warmup_steps"]) >= int(
        pointer_one_shot["max_steps"]
    ):
        raise ConfigError(
            "'pointer_route.one_shot.warmup_steps' must be less than max_steps"
        )
    if pointer_one_shot.get("precision") not in {"bf16", "fp16"}:
        raise ConfigError(
            "'pointer_route.one_shot.precision' must be bf16 or fp16"
        )
    for key in (
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "route_soft_ce_weight",
        "route_hard_ce_weight",
        "local_map_cosine_weight",
        "identity_map_cosine_weight",
    ):
        _require_real(pointer_one_shot, key, minimum=0.0)
    if float(pointer_one_shot["route_soft_ce_weight"]) <= 0.0:
        raise ConfigError(
            "'pointer_route.one_shot.route_soft_ce_weight' must be positive"
        )
    one_shot_gates = _require_mapping(pointer_one_shot, "gates")
    _require_real(
        one_shot_gates,
        "max_hard_auc_drop_vs_quality",
        minimum=0.0,
        maximum=1.0,
    )
    _require_real(
        one_shot_gates,
        "min_oracle_gap_recovery",
        minimum=0.0,
        maximum=1.0,
    )
    _require_int(one_shot_gates, "min_hard_scenario_wins")
    _require_real(
        one_shot_gates,
        "max_permutation_logit_difference",
        minimum=0.0,
    )
    diffusion = _require_mapping(pointer_route, "diffusion")
    if diffusion.get("protocol") != "plain_confidence":
        raise ConfigError(
            "'pointer_route.diffusion.protocol' must be 'plain_confidence'"
        )
    _require_bool(diffusion, "paired_force_full_steps")
    all_mask_probability = _require_real(
        diffusion,
        "all_mask_probability",
        minimum=0.0,
        maximum=1.0,
    )
    random_partial_probability = _require_real(
        diffusion,
        "random_partial_probability",
        minimum=0.0,
        maximum=1.0,
    )
    if abs(all_mask_probability + random_partial_probability - 1.0) > 1e-8:
        raise ConfigError(
            "pointer-route diffusion corruption probabilities must sum to 1"
        )
    _require_real(
        diffusion,
        "min_mask_ratio",
        minimum=1e-8,
        maximum=1.0,
    )
    default_route_steps = _require_int(diffusion, "default_steps")
    route_evaluation_steps = diffusion.get("evaluation_steps")
    if (
        not isinstance(route_evaluation_steps, list)
        or not route_evaluation_steps
        or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) <= 0
            for value in route_evaluation_steps
        )
    ):
        raise ConfigError(
            "'pointer_route.diffusion.evaluation_steps' must contain "
            "positive integers"
        )
    if len({int(value) for value in route_evaluation_steps}) != len(
        route_evaluation_steps
    ):
        raise ConfigError(
            "'pointer_route.diffusion.evaluation_steps' must be unique"
        )
    if default_route_steps not in {
        int(value) for value in route_evaluation_steps
    }:
        raise ConfigError(
            "'pointer_route.diffusion.default_steps' must be evaluated"
        )
    if diffusion.get("schedule") != "cosine":
        raise ConfigError(
            "'pointer_route.diffusion.schedule' must be 'cosine'"
        )
    diffusion_gates = _require_mapping(diffusion, "gates")
    for key in (
        "min_four_step_gain_over_one_step",
        "min_four_step_gain_over_paired_one_shot",
        "max_clean_auc_drop_vs_one_step",
        "max_permutation_map_difference",
    ):
        _require_real(diffusion_gates, key, minimum=0.0)
    _require_int(diffusion_gates, "min_hard_scenario_step_wins")
    _require_bool(diffusion_gates, "require_positive_net_correction")
    rollout_rescue = _require_mapping(diffusion, "rollout_rescue")
    if (
        rollout_rescue.get("protocol")
        != "replace_random_partial_with_self_rollout"
    ):
        raise ConfigError(
            "'pointer_route.diffusion.rollout_rescue.protocol' must be "
            "'replace_random_partial_with_self_rollout'"
        )
    rollout_all_mask_probability = _require_real(
        rollout_rescue,
        "all_mask_probability",
        minimum=0.0,
        maximum=1.0,
    )
    rollout_probability = _require_real(
        rollout_rescue,
        "rollout_probability",
        minimum=0.0,
        maximum=1.0,
    )
    if (
        abs(rollout_all_mask_probability + rollout_probability - 1.0)
        > 1e-8
    ):
        raise ConfigError(
            "pointer-route rollout rescue corruption probabilities "
            "must sum to 1"
        )
    rollout_total_steps = _require_int(
        rollout_rescue,
        "rollout_total_steps",
        minimum=2,
    )
    rollout_depths = rollout_rescue.get("rollout_depths")
    if (
        not isinstance(rollout_depths, list)
        or not rollout_depths
        or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 1 <= int(value) < rollout_total_steps
            for value in rollout_depths
        )
    ):
        raise ConfigError(
            "'pointer_route.diffusion.rollout_rescue.rollout_depths' "
            "must contain steps in [1, rollout_total_steps)"
        )
    if len({int(value) for value in rollout_depths}) != len(
        rollout_depths
    ):
        raise ConfigError(
            "'pointer_route.diffusion.rollout_rescue.rollout_depths' "
            "must be unique"
        )
    if rollout_rescue.get("rollout_depth_sampling") != "uniform":
        raise ConfigError(
            "'pointer_route.diffusion.rollout_rescue."
            "rollout_depth_sampling' must be 'uniform'"
        )
    for key in (
        "stop_gradient",
        "use_gpu_resident_training_cache",
        "evidence_used",
        "remask_used",
    ):
        _require_bool(rollout_rescue, key)
    if rollout_rescue["stop_gradient"] is not True:
        raise ConfigError(
            "pointer-route rollout rescue must stop gradients through rollout"
        )
    if rollout_rescue["use_gpu_resident_training_cache"] is not True:
        raise ConfigError(
            "pointer-route rollout rescue requires the GPU-resident cache"
        )
    if rollout_rescue["evidence_used"] or rollout_rescue["remask_used"]:
        raise ConfigError(
            "pointer-route rollout rescue cannot use evidence or remasking"
        )
    rollout_gates = _require_mapping(rollout_rescue, "gates")
    for key in (
        "min_four_step_gain_over_one_step",
        "min_four_step_gain_over_paired_one_shot",
        "min_four_step_gain_over_plain_d0",
        "max_clean_auc_drop_vs_one_step",
        "max_permutation_map_difference",
    ):
        _require_real(rollout_gates, key, minimum=0.0)
    _require_int(rollout_gates, "min_hard_scenario_step_wins")
    _require_bool(rollout_gates, "require_positive_net_correction")
    evidence_order = _require_mapping(diffusion, "evidence_order")
    if evidence_order.get("protocol") != "two_level_commit_order_only":
        raise ConfigError(
            "'pointer_route.diffusion.evidence_order.protocol' must be "
            "'two_level_commit_order_only'"
        )
    evidence_lambdas = evidence_order.get("lambda_candidates")
    if (
        not isinstance(evidence_lambdas, list)
        or not evidence_lambdas
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or float(value) <= 0.0
            for value in evidence_lambdas
        )
    ):
        raise ConfigError(
            "'pointer_route.diffusion.evidence_order.lambda_candidates' "
            "must contain positive numbers"
        )
    if len({float(value) for value in evidence_lambdas}) != len(
        evidence_lambdas
    ):
        raise ConfigError(
            "'pointer_route.diffusion.evidence_order.lambda_candidates' "
            "must be unique"
        )
    for key in (
        "auc_tie_tolerance",
        "identity_neighbor_fraction",
        "identity_temperature",
        "quality_weight",
        "local_temperature",
        "reliability_floor",
    ):
        minimum = 1e-12 if key == "reliability_floor" else 0.0
        maximum = 1.0 if key == "identity_neighbor_fraction" else None
        _require_real(
            evidence_order,
            key,
            minimum=minimum,
            maximum=maximum,
        )
    if float(evidence_order["identity_neighbor_fraction"]) <= 0.0:
        raise ConfigError(
            "pointer-route evidence identity_neighbor_fraction "
            "must be positive"
        )
    for key in ("stop_gradient", "modify_route_logits", "remask_used"):
        _require_bool(evidence_order, key)
    if evidence_order["stop_gradient"] is not True:
        raise ConfigError(
            "pointer-route evidence ordering must use stop-gradient evidence"
        )
    if evidence_order["modify_route_logits"] or evidence_order["remask_used"]:
        raise ConfigError(
            "pointer-route evidence ordering cannot modify logits or remask"
        )
    evidence_gates = _require_mapping(evidence_order, "gates")
    for key in (
        "min_gain_over_confidence_4step",
        "min_gain_over_rollout_1step",
        "max_clean_auc_drop",
        "max_permutation_map_difference",
    ):
        _require_real(evidence_gates, key, minimum=0.0)
    _require_int(evidence_gates, "min_hard_scenario_wins")
    _require_bool(evidence_gates, "require_positive_net_correction")
    evidence_logits = _require_mapping(diffusion, "evidence_logits")
    if evidence_logits.get("protocol") != "two_level_logits_only":
        raise ConfigError(
            "'pointer_route.diffusion.evidence_logits.protocol' must be "
            "'two_level_logits_only'"
        )
    logits_lambdas = evidence_logits.get("lambda_candidates")
    if (
        not isinstance(logits_lambdas, list)
        or not logits_lambdas
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or float(value) <= 0.0
            for value in logits_lambdas
        )
    ):
        raise ConfigError(
            "'pointer_route.diffusion.evidence_logits.lambda_candidates' "
            "must contain positive numbers"
        )
    if len({float(value) for value in logits_lambdas}) != len(
        logits_lambdas
    ):
        raise ConfigError(
            "'pointer_route.diffusion.evidence_logits.lambda_candidates' "
            "must be unique"
        )
    for key in (
        "auc_tie_tolerance",
        "identity_neighbor_fraction",
        "identity_temperature",
        "quality_weight",
        "local_temperature",
        "reliability_floor",
    ):
        minimum = 1e-12 if key == "reliability_floor" else 0.0
        maximum = 1.0 if key == "identity_neighbor_fraction" else None
        _require_real(
            evidence_logits,
            key,
            minimum=minimum,
            maximum=maximum,
        )
    if float(evidence_logits["identity_neighbor_fraction"]) <= 0.0:
        raise ConfigError(
            "pointer-route evidence logits identity_neighbor_fraction "
            "must be positive"
        )
    for key in (
        "stop_gradient",
        "modify_selection_logits",
        "extra_order_guidance",
        "remask_used",
    ):
        _require_bool(evidence_logits, key)
    if evidence_logits["stop_gradient"] is not True:
        raise ConfigError(
            "pointer-route evidence logits must use stop-gradient evidence"
        )
    if evidence_logits["modify_selection_logits"] is not True:
        raise ConfigError(
            "pointer-route evidence logits must modify selection logits"
        )
    if (
        evidence_logits["extra_order_guidance"]
        or evidence_logits["remask_used"]
    ):
        raise ConfigError(
            "pointer-route evidence logits cannot add order guidance or remask"
        )
    logits_gates = _require_mapping(evidence_logits, "gates")
    for key in (
        "min_four_step_gain_over_guided_one_step",
        "min_four_step_gain_over_confidence_one_step",
        "max_clean_auc_drop_vs_guided_one_step",
        "max_permutation_map_difference",
    ):
        _require_real(logits_gates, key, minimum=0.0)
    _require_int(logits_gates, "min_hard_scenario_step_wins")
    for key in (
        "require_positive_net_correction",
        "require_wrong_identity_selection_reduction",
    ):
        _require_bool(logits_gates, key)

    evidence_remask = _require_mapping(diffusion, "evidence_remask")
    if (
        evidence_remask.get("protocol")
        != "calibration_tuned_budgeted_remask"
    ):
        raise ConfigError(
            "'pointer_route.diffusion.evidence_remask.protocol' must be "
            "'calibration_tuned_budgeted_remask'"
        )
    if evidence_remask.get("tuning_split") != "route_calibration":
        raise ConfigError(
            "pointer-route remask tuning must use route_calibration"
        )
    if evidence_remask.get("evaluation_split") != "route_validation":
        raise ConfigError(
            "pointer-route remask evaluation must use route_validation"
        )
    remask_lambdas = evidence_remask.get("lambda_candidates")
    if (
        not isinstance(remask_lambdas, list)
        or not remask_lambdas
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or float(value) <= 0.0
            for value in remask_lambdas
        )
    ):
        raise ConfigError(
            "'pointer_route.diffusion.evidence_remask.lambda_candidates' "
            "must contain positive numbers"
        )
    if len({float(value) for value in remask_lambdas}) != len(
        remask_lambdas
    ):
        raise ConfigError(
            "'pointer_route.diffusion.evidence_remask.lambda_candidates' "
            "must be unique"
        )
    recipes = evidence_remask.get("evidence_recipes")
    if not isinstance(recipes, list) or not recipes:
        raise ConfigError(
            "'pointer_route.diffusion.evidence_remask.evidence_recipes' "
            "must be a non-empty list"
        )
    recipe_names: list[str] = []
    for index, recipe in enumerate(recipes):
        if not isinstance(recipe, Mapping):
            raise ConfigError(
                "pointer-route remask evidence recipes must be mappings"
            )
        name = recipe.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(
                "pointer-route remask evidence recipes require names"
            )
        recipe_names.append(name)
        for key in (
            "identity_neighbor_fraction",
            "identity_temperature",
            "quality_weight",
            "local_temperature",
        ):
            if key not in recipe:
                raise ConfigError(
                    "pointer-route remask evidence recipe "
                    f"{index} is missing '{key}'"
                )
            minimum = 0.0
            maximum = (
                1.0 if key == "identity_neighbor_fraction" else None
            )
            _require_real(
                recipe,
                key,
                minimum=minimum,
                maximum=maximum,
            )
        if float(recipe["identity_neighbor_fraction"]) <= 0.0:
            raise ConfigError(
                "pointer-route remask identity_neighbor_fraction "
                "must be positive"
            )
    if len(set(recipe_names)) != len(recipe_names):
        raise ConfigError(
            "pointer-route remask evidence recipe names must be unique"
        )
    _require_real(
        evidence_remask,
        "auc_tie_tolerance",
        minimum=0.0,
    )
    _require_real(
        evidence_remask,
        "reliability_floor",
        minimum=1e-12,
    )
    for key, maximum in (
        ("remask_fraction_candidates", 1.0),
        ("remask_min_score_gain_candidates", None),
    ):
        values = evidence_remask.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or float(value) < 0.0
                or (maximum is not None and float(value) > maximum)
                for value in values
            )
        ):
            raise ConfigError(
                f"'pointer_route.diffusion.evidence_remask.{key}' "
                "contains invalid values"
            )
        if key == "remask_fraction_candidates" and any(
            float(value) <= 0.0 for value in values
        ):
            raise ConfigError(
                "pointer-route remask fractions must be positive"
            )
        if len({float(value) for value in values}) != len(values):
            raise ConfigError(
                f"'pointer_route.diffusion.evidence_remask.{key}' "
                "must be unique"
            )
    for key in (
        "stop_gradient",
        "modify_selection_logits",
        "bounded_reversible_commit",
        "final_step_full_commit",
    ):
        _require_bool(evidence_remask, key)
        if evidence_remask[key] is not True:
            raise ConfigError(
                f"pointer-route evidence remask requires '{key}'"
            )
    remask_gates = _require_mapping(evidence_remask, "gates")
    for key in (
        "min_four_step_gain_over_tuned_no_remask",
        "min_four_step_gain_over_confidence_one_step",
        "max_clean_auc_drop_vs_tuned_no_remask",
        "max_permutation_map_difference",
    ):
        _require_real(remask_gates, key, minimum=0.0)
    _require_int(remask_gates, "min_hard_scenario_wins")
    for key in (
        "require_positive_net_correction",
        "require_actual_remask",
        "require_remask_ecr_exceeds_eir",
    ):
        _require_bool(remask_gates, key)

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
