"""Build the collision-safe Stage 16 expanded-data holdout report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALITY = "quality_pooling"
P1_1 = "continuous_local_router"
P1_2 = "identity_gated_anchor_residual_router"
P1_3 = "bounded_scalar_evidence_router"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(materialized[0]),
        )
        writer.writeheader()
        writer.writerows(materialized)


def _find_method(
    rows: Iterable[Mapping[str, str]],
    *,
    split: str,
    method: str,
) -> dict[str, str]:
    matches = [
        dict(row)
        for row in rows
        if row["split"] == split and row["method"] == method
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {split}/{method} row, found {len(matches)}"
        )
    return matches[0]


def _normalized_core(
    row: Mapping[str, str],
    *,
    stage: str,
) -> dict[str, Any]:
    return {
        "split": row["split"],
        "stage": stage,
        "method": row["method"],
        "pooled_all_roc_auc": float(row["pooled_all_roc_auc"]),
        "hard_macro_roc_auc": float(row["hard_macro_roc_auc"]),
        "eer": float(row["eer"]),
        "tar_at_far_1e-2": float(row["tar_at_far_1e-2"]),
        "tar_at_far_1e-3": float(row["tar_at_far_1e-3"]),
        "rank1_accuracy": float(row["rank1_accuracy"]),
        "margin": float(
            row.get("margin")
            or row.get("mean_genuine_impostor_margin")
            or "nan"
        ),
        "teacher_map_cosine": float(row["teacher_map_cosine"]),
    }


def _metric_delta(
    learned: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> dict[str, float]:
    keys = (
        "pooled_all_roc_auc",
        "hard_macro_roc_auc",
        "eer",
        "tar_at_far_1e-2",
        "tar_at_far_1e-3",
        "rank1_accuracy",
        "margin",
        "teacher_map_cosine",
    )
    return {
        key: float(learned[key]) - float(quality[key])
        for key in keys
    }


def build_report(
    *,
    output_root: Path,
    old_split_root: Path,
    expanded_split_root: Path,
    feature_cache_root: Path,
    set_root: Path,
    quantization_root: Path,
    p1_1_root: Path,
    p1_2_root: Path,
    p1_3_root: Path,
    old_p1_3_root: Path | None,
) -> dict[str, str]:
    output_root = output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"report output already exists and is not empty: {output_root}"
        )
    artifact_root = output_root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=False)

    old_split = _read_json(old_split_root / "manifest.json")
    expanded_split = _read_json(expanded_split_root / "manifest.json")
    feature_cache = _read_json(feature_cache_root / "manifest.json")
    sets = _read_json(set_root / "manifest.json")
    quantization = _read_json(quantization_root / "manifest.json")
    p1_1_manifest = _read_json(p1_1_root / "logs" / "run_manifest.json")
    p1_2_manifest = _read_json(p1_2_root / "logs" / "run_manifest.json")
    p1_3_manifest = _read_json(p1_3_root / "logs" / "run_manifest.json")

    p1_1_rows = _read_csv(p1_1_root / "artifacts" / "core_comparison.csv")
    p1_2_rows = _read_csv(p1_2_root / "artifacts" / "core_comparison.csv")
    p1_3_rows = _read_csv(p1_3_root / "artifacts" / "core_comparison.csv")
    core = [
        _normalized_core(
            _find_method(p1_1_rows, split="val", method=QUALITY),
            stage="baseline",
        ),
        _normalized_core(
            _find_method(p1_1_rows, split="val", method=P1_1),
            stage="P1-1",
        ),
        _normalized_core(
            _find_method(p1_2_rows, split="val", method=P1_2),
            stage="P1-2",
        ),
        _normalized_core(
            _find_method(p1_3_rows, split="val", method=P1_3),
            stage="P1-3",
        ),
        _normalized_core(
            _find_method(p1_3_rows, split="test", method=QUALITY),
            stage="baseline",
        ),
        _normalized_core(
            _find_method(p1_3_rows, split="test", method=P1_3),
            stage="P1-3 locked",
        ),
    ]
    core_path = artifact_root / "core_comparison.csv"
    _write_csv(core_path, core)

    scenario_rows = [
        row
        for row in _read_csv(
            p1_3_root / "artifacts" / "scenario_metrics.csv"
        )
        if row["method"] in {QUALITY, P1_3}
    ]
    scenario_path = artifact_root / "scenario_metrics.csv"
    _write_csv(scenario_path, scenario_rows)

    val_quality = next(
        row
        for row in core
        if row["split"] == "val" and row["method"] == QUALITY
    )
    val_p1_3 = next(
        row
        for row in core
        if row["split"] == "val" and row["method"] == P1_3
    )
    test_quality = next(
        row
        for row in core
        if row["split"] == "test" and row["method"] == QUALITY
    )
    test_p1_3 = next(
        row
        for row in core
        if row["split"] == "test" and row["method"] == P1_3
    )
    test_delta = _metric_delta(test_p1_3, test_quality)

    old_overlap: dict[str, int] = {}
    all_old_ids: set[int] = set()
    all_expanded_ids: set[int] = set()
    for split in ("train", "val", "test"):
        with (old_split_root / "identities.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            old_ids = {
                int(row["identity_id"])
                for row in csv.DictReader(handle)
                if row["split"] == split
            }
        with (expanded_split_root / "identities.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            expanded_ids = {
                int(row["identity_id"])
                for row in csv.DictReader(handle)
                if row["split"] == split
            }
        old_overlap[split] = len(old_ids & expanded_ids)
        all_old_ids.update(old_ids)
        all_expanded_ids.update(expanded_ids)
    old_overlap["global"] = len(all_old_ids & all_expanded_ids)

    test_scenarios: list[dict[str, Any]] = []
    for scenario in (
        "clean",
        "low_quality",
        "complementary_occlusion",
        "common_occlusion",
        "wrong_identity",
    ):
        quality_row = next(
            row
            for row in scenario_rows
            if row["split"] == "test"
            and row["method"] == QUALITY
            and row["scenario"] == scenario
        )
        learned_row = next(
            row
            for row in scenario_rows
            if row["split"] == "test"
            and row["method"] == P1_3
            and row["scenario"] == scenario
        )
        test_scenarios.append(
            {
                "scenario": scenario,
                "quality_auc": float(quality_row["roc_auc"]),
                "p1_3_auc": float(learned_row["roc_auc"]),
                "auc_delta": (
                    float(learned_row["roc_auc"])
                    - float(quality_row["roc_auc"])
                ),
            }
        )

    old_p1_3_summary = None
    if old_p1_3_root is not None:
        old_rows = _read_csv(
            old_p1_3_root / "artifacts" / "core_comparison.csv"
        )
        old_quality = _normalized_core(
            _find_method(old_rows, split="val", method=QUALITY),
            stage="old baseline",
        )
        old_learned = _normalized_core(
            _find_method(old_rows, split="val", method=P1_3),
            stage="old P1-3",
        )
        old_p1_3_summary = {
            "quality": old_quality,
            "p1_3": old_learned,
            "delta": _metric_delta(old_learned, old_quality),
        }

    audit = {
        "schema_version": 1,
        "protocol": {
            "identity_source": "CelebA official identity/partition annotations",
            "old_reference_split": str(old_split_root.resolve()),
            "expanded_split": str(expanded_split_root.resolve()),
            "old_identity_overlap": old_overlap,
            "intermediate_test_policy": "P1-1 and P1-2 validation-only",
            "final_test_policy": (
                "one locked P1-3 checkpoint selected on validation and "
                "evaluated once"
            ),
        },
        "scale": {
            "old": old_split["selection"]["splits"],
            "expanded": expanded_split["selection"]["splits"],
            "feature_cache": {
                split: feature_cache["splits"][split]["images"]
                for split in ("train", "val", "test")
            },
            "sets": {
                split: sets["splits"][split]["sets"]
                for split in ("train", "val", "test")
            },
        },
        "quantization": {
            "projection": quantization["projection"],
            "codebook": quantization["codebook"],
            "metrics": quantization["metrics"],
        },
        "runs": {
            "p1_1": {
                "root": str(p1_1_root.resolve()),
                "status": p1_1_manifest["status"],
                "test_dataset_constructed": p1_1_manifest[
                    "test_dataset_constructed"
                ],
            },
            "p1_2": {
                "root": str(p1_2_root.resolve()),
                "status": p1_2_manifest["status"],
                "test_dataset_constructed": p1_2_manifest[
                    "test_dataset_constructed"
                ],
            },
            "p1_3": {
                "root": str(p1_3_root.resolve()),
                "status": p1_3_manifest["status"],
                "test_dataset_constructed": p1_3_manifest[
                    "test_dataset_constructed"
                ],
                "selection": p1_3_manifest["selection"],
                "test_gate": p1_3_manifest["test_gate"],
                "decision": p1_3_manifest["decision"],
            },
        },
        "metrics": {
            "validation_delta_p1_3_vs_quality": _metric_delta(
                val_p1_3,
                val_quality,
            ),
            "test_delta_p1_3_vs_quality": test_delta,
            "test_scenarios": test_scenarios,
            "old_pilot_validation": old_p1_3_summary,
        },
        "caveat": (
            "This is one locked holdout evaluation. No confidence interval "
            "or repeated-split evidence was computed, so no statistical "
            "significance claim is made."
        ),
    }
    audit_path = artifact_root / "scale_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    scenario_lines = "\n".join(
        "| {scenario} | {quality_auc:.6f} | {p1_3_auc:.6f} | "
        "{auc_delta:+.6f} |".format(**row)
        for row in test_scenarios
    )
    report = f"""# Stage 16：扩大训练数据与全新身份隔离 Holdout

## 结论

本阶段已完成实际数据扩容、重拟合、三段路由训练和一次锁定 holdout
评估。新 split 为 **1000 train / 100 val / 250 test identities**，
分别含 **23342 / 2444 / 5610 images**；与旧 pilot 的对应身份及全局身份
重叠均为 0。P1-1、P1-2 强制 validation-only，只有 validation 锁定的
P1-3 `scalar_frozen@550` 在新 test 上评估一次。

最终结论为 **严格 No-Go、接近门槛**。P1-3 在 test 上将 hard macro AUC
从 {test_quality['hard_macro_roc_auc']:.6f} 提升到
{test_p1_3['hard_macro_roc_auc']:.6f}
（{test_delta['hard_macro_roc_auc']:+.6f}），略低于预注册
`+0.002` 门槛 {0.002 - test_delta['hard_macro_roc_auc']:.6f}。它显著改善
互补遮挡，但在 low-quality 与 wrong-ID 上没有保持 validation 中的正增益。

## 规模与隔离

| split | 旧 identities | 新 identities | 新 images | 新 sets | 与旧身份重叠 |
|---|---:|---:|---:|---:|---:|
| train | {old_split['selection']['splits']['train']['identities']} | {expanded_split['selection']['splits']['train']['identities']} | {expanded_split['selection']['splits']['train']['images']} | {sets['splits']['train']['sets']} | {old_overlap['train']} |
| val | {old_split['selection']['splits']['val']['identities']} | {expanded_split['selection']['splits']['val']['identities']} | {expanded_split['selection']['splits']['val']['images']} | {sets['splits']['val']['sets']} | {old_overlap['val']} |
| test | {old_split['selection']['splits']['test']['identities']} | {expanded_split['selection']['splits']['test']['identities']} | {expanded_split['selection']['splits']['test']['images']} | {sets['splits']['test']['sets']} | {old_overlap['test']} |

## 核心 test 指标

| 方法 | pooled AUC | hard AUC | EER | TAR@1e-2 | TAR@1e-3 | Rank-1 | margin | map cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| quality pooling | {test_quality['pooled_all_roc_auc']:.6f} | {test_quality['hard_macro_roc_auc']:.6f} | {test_quality['eer']:.6f} | {test_quality['tar_at_far_1e-2']:.4f} | {test_quality['tar_at_far_1e-3']:.4f} | {test_quality['rank1_accuracy']:.4f} | {test_quality['margin']:.6f} | {test_quality['teacher_map_cosine']:.6f} |
| P1-3 scalar evidence | {test_p1_3['pooled_all_roc_auc']:.6f} | {test_p1_3['hard_macro_roc_auc']:.6f} | {test_p1_3['eer']:.6f} | {test_p1_3['tar_at_far_1e-2']:.4f} | {test_p1_3['tar_at_far_1e-3']:.4f} | {test_p1_3['rank1_accuracy']:.4f} | {test_p1_3['margin']:.6f} | {test_p1_3['teacher_map_cosine']:.6f} |
| delta | {test_delta['pooled_all_roc_auc']:+.6f} | {test_delta['hard_macro_roc_auc']:+.6f} | {test_delta['eer']:+.6f} | {test_delta['tar_at_far_1e-2']:+.4f} | {test_delta['tar_at_far_1e-3']:+.4f} | {test_delta['rank1_accuracy']:+.4f} | {test_delta['margin']:+.6f} | {test_delta['teacher_map_cosine']:+.6f} |

## Test 分场景 AUC

| 场景 | quality | P1-3 | delta |
|---|---:|---:|---:|
{scenario_lines}

## Validation 路由演进

| 方法 | pooled AUC | hard AUC | EER | Rank-1 | map cosine |
|---|---:|---:|---:|---:|---:|
| quality | {core[0]['pooled_all_roc_auc']:.6f} | {core[0]['hard_macro_roc_auc']:.6f} | {core[0]['eer']:.6f} | {core[0]['rank1_accuracy']:.4f} | {core[0]['teacher_map_cosine']:.6f} |
| P1-1 | {core[1]['pooled_all_roc_auc']:.6f} | {core[1]['hard_macro_roc_auc']:.6f} | {core[1]['eer']:.6f} | {core[1]['rank1_accuracy']:.4f} | {core[1]['teacher_map_cosine']:.6f} |
| P1-2 | {core[2]['pooled_all_roc_auc']:.6f} | {core[2]['hard_macro_roc_auc']:.6f} | {core[2]['eer']:.6f} | {core[2]['rank1_accuracy']:.4f} | {core[2]['teacher_map_cosine']:.6f} |
| P1-3 | {core[3]['pooled_all_roc_auc']:.6f} | {core[3]['hard_macro_roc_auc']:.6f} | {core[3]['eer']:.6f} | {core[3]['rank1_accuracy']:.4f} | {core[3]['teacher_map_cosine']:.6f} |

## 量化审计

- PCA：512→128，训练 token 拟合，解释方差
  {quantization['projection']['explained_variance_ratio_sum']:.4f}。
- K=1024，训练 codebook utilization
  {quantization['metrics']['train']['codebook_utilization']:.4f}，
  val/test 为
  {quantization['metrics']['val']['codebook_utilization']:.4f}/
  {quantization['metrics']['test']['codebook_utilization']:.4f}。
- val/test mean token cosine 仅
  {quantization['metrics']['val']['mean_token_cosine']:.4f}/
  {quantization['metrics']['test']['mean_token_cosine']:.4f}，离散重建仍有
  明显信息损失，不能据此宣布离散扩散已可行。

## 研究判断

扩大数据后，P1-3 的 validation hard 增益从旧 pilot 的约 `+0.00035`
提高到 `{audit['metrics']['validation_delta_p1_3_vs_quality']['hard_macro_roc_auc']:+.6f}`，
并在全新 test 上保留 `{test_delta['hard_macro_roc_auc']:+.6f}`。这说明
学习式 evidence routing 并非完全无效，且优势高度集中于互补遮挡。
但 low-quality 与 wrong-ID 的 test 退化表明路由证据尚未跨身份稳定泛化。

因此不应继续按 test 调 gate 或放宽门槛。建议把当前 checkpoint 作为
冻结参考，并把下一步限定为：在新的 identity-disjoint validation split
上改进 wrong-ID/low-quality 证据，再预注册另一套未见身份 holdout；若不能
同时修复这两类场景，则停止 late-hook router，转向更早层或显式
outlier rejection。

本报告来自一次锁定 holdout 评估，未计算重复划分或置信区间，**不声称统计
显著性**。
"""
    report_path = artifact_root / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "status": "passed",
        "stage": "stage16-expanded-holdout-report",
        "inputs": {
            "old_split_root": str(old_split_root.resolve()),
            "expanded_split_root": str(expanded_split_root.resolve()),
            "feature_cache_root": str(feature_cache_root.resolve()),
            "set_root": str(set_root.resolve()),
            "quantization_root": str(quantization_root.resolve()),
            "p1_1_root": str(p1_1_root.resolve()),
            "p1_2_root": str(p1_2_root.resolve()),
            "p1_3_root": str(p1_3_root.resolve()),
        },
        "artifacts": {
            "report": str(report_path),
            "core_comparison_csv": str(core_path),
            "scenario_metrics_csv": str(scenario_path),
            "scale_audit_json": str(audit_path),
        },
        "decision": p1_3_manifest["decision"],
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return {
        "report": str(report_path),
        "core_comparison_csv": str(core_path),
        "scenario_metrics_csv": str(scenario_path),
        "scale_audit_json": str(audit_path),
        "manifest": str(manifest_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "stage16-expanded-holdout-report",
    )
    parser.add_argument(
        "--old-split-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "real"
            / "celeba"
            / "splits"
            / "pilot-seed20260725"
        ),
    )
    parser.add_argument(
        "--expanded-split-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "real"
            / "celeba"
            / "splits"
            / "expanded-seed20260725"
        ),
    )
    parser.add_argument(
        "--feature-cache-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "cache"
            / "real_features"
            / "stage16-expanded-feature-cache"
        ),
    )
    parser.add_argument(
        "--set-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "real_sets"
            / "stage16-expanded-real-sets"
        ),
    )
    parser.add_argument(
        "--quantization-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "cache"
            / "quantization"
            / "stage16-expanded-quantization"
        ),
    )
    parser.add_argument(
        "--p1-1-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "stage16-p1-1-expanded",
    )
    parser.add_argument(
        "--p1-2-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "stage16-p1-2-expanded",
    )
    parser.add_argument(
        "--p1-3-root",
        type=Path,
        default=(
            PROJECT_ROOT / "outputs" / "stage16-p1-3-expanded-holdout"
        ),
    )
    parser.add_argument(
        "--old-p1-3-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "p1-3-scalar-evidence-router-pilot-reviewed"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifacts = build_report(
        output_root=args.output_root,
        old_split_root=args.old_split_root,
        expanded_split_root=args.expanded_split_root,
        feature_cache_root=args.feature_cache_root,
        set_root=args.set_root,
        quantization_root=args.quantization_root,
        p1_1_root=args.p1_1_root,
        p1_2_root=args.p1_2_root,
        p1_3_root=args.p1_3_root,
        old_p1_3_root=args.old_p1_3_root,
    )
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
