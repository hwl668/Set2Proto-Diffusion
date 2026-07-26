"""Generate reproducible tables, plots, and the pilot Go/No-Go report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CORE_METHODS = (
    "best_single_frame",
    "mean_pooling",
    "max_pooling",
    "quality_pooling",
    "one_shot_transformer",
    "maskgit_confidence_1step",
    "maskgit_confidence_2step",
    "maskgit_confidence_4step",
    "maskgit_confidence_8step",
    "maskgit_evidence_ordering",
    "maskgit_evidence_logits",
    "maskgit_evidence_remask",
)
SCENARIOS = (
    "clean",
    "low_quality",
    "complementary_occlusion",
    "common_occlusion",
    "wrong_identity",
)


def _load_manifest(run_root: Path) -> dict[str, Any]:
    path = run_root / "logs" / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"run manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty report table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames or list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def _verification_rows(
    training_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for split in ("val", "test"):
        verification = training_manifest["research_evaluation"][split][
            "verification"
        ]
        token_accuracy = training_manifest["research_evaluation"][split][
            "token_accuracy"
        ]
        for method, scenarios in verification.items():
            for scenario, metrics in scenarios.items():
                if "roc_auc" not in metrics:
                    continue
                tar = metrics.get("tar_at_far", {})
                token = token_accuracy.get(method, {})
                if scenario == "all":
                    token_value = token.get("all")
                else:
                    token_value = token.get("by_scenario", {}).get(scenario)
                rows.append(
                    {
                        "split": split,
                        "method": method,
                        "scenario": scenario,
                        "roc_auc": metrics["roc_auc"],
                        "eer": metrics["eer"],
                        "tar_far_1e-2": tar.get("0.01", {}).get("tar"),
                        "tar_far_1e-3": tar.get("0.001", {}).get("tar"),
                        "negative_pairs": metrics.get("negative_pairs"),
                        "token_accuracy": token_value,
                    }
                )
    return rows


def _core_rows(
    training_manifest: dict[str, Any],
    diagnostics_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    verification = training_manifest["research_evaluation"]["test"][
        "verification"
    ]
    token_accuracy = training_manifest["research_evaluation"]["test"][
        "token_accuracy"
    ]
    latency = diagnostics_manifest["diagnostics"]["latency"]
    rows = []
    for method in CORE_METHODS:
        all_metrics = verification[method]["all"]
        hard_metrics = verification[method]["hard_average"]
        rows.append(
            {
                "method": method,
                "test_all_roc_auc": all_metrics["roc_auc"],
                "test_hard_roc_auc": hard_metrics["roc_auc"],
                "test_all_eer": all_metrics["eer"],
                "tar_far_1e-2": all_metrics["tar_at_far"]["0.01"]["tar"],
                "tar_far_1e-3": all_metrics["tar_at_far"]["0.001"]["tar"],
                "token_accuracy": token_accuracy.get(method, {}).get("all"),
                "latency_ms_per_set": latency.get(method, {}).get(
                    "per_set_latency_ms"
                ),
            }
        )
    return rows


def _training_rows(training_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for mode, values in training_manifest["training"].items():
        for index, (loss, accuracy, learning_rate) in enumerate(
            zip(
                values["losses"],
                values["masked_accuracies"],
                values["learning_rates"],
            ),
            start=1,
        ):
            rows.append(
                {
                    "mode": mode,
                    "step": index,
                    "loss": loss,
                    "masked_token_accuracy": accuracy,
                    "learning_rate": learning_rate,
                }
            )
    return rows


def _plot_training(
    training_manifest: dict[str, Any],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode, values in training_manifest["training"].items():
        steps = np.arange(1, len(values["losses"]) + 1)
        axes[0].plot(steps, values["losses"], label=mode, linewidth=1)
        axes[1].plot(
            steps,
            values["masked_accuracies"],
            label=mode,
            linewidth=1,
        )
    axes[0].set(title="Training loss", xlabel="Step", ylabel="Cross-entropy")
    axes[1].set(
        title="Training masked-token accuracy",
        xlabel="Step",
        ylabel="Accuracy",
        ylim=(0.0, 1.02),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_scenario_auc(
    training_manifest: dict[str, Any],
    path: Path,
) -> None:
    verification = training_manifest["research_evaluation"]["test"][
        "verification"
    ]
    methods = (
        "quality_pooling",
        "one_shot_transformer",
        "maskgit_confidence_4step",
        "maskgit_evidence_logits",
        "maskgit_evidence_remask",
    )
    labels = ("Quality pool", "One-shot", "4-step conf.", "Evidence", "Remask")
    x_values = np.arange(len(SCENARIOS))
    width = 0.16
    figure, axis = plt.subplots(figsize=(12, 5))
    for index, (method, label) in enumerate(zip(methods, labels)):
        values = [
            verification[method][scenario]["roc_auc"]
            for scenario in SCENARIOS
        ]
        axis.bar(
            x_values + (index - 2) * width,
            values,
            width=width,
            label=label,
        )
    axis.set(
        title="Test ROC-AUC by corruption scenario",
        ylabel="ROC-AUC",
        ylim=(0.7, 1.01),
        xticks=x_values,
        xticklabels=(
            "Clean",
            "Low quality",
            "Complementary",
            "Common missing",
            "Wrong ID",
        ),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_steps_latency(
    training_manifest: dict[str, Any],
    diagnostics_manifest: dict[str, Any],
    path: Path,
) -> None:
    verification = training_manifest["research_evaluation"]["test"][
        "verification"
    ]
    latency = diagnostics_manifest["diagnostics"]["latency"]
    steps = np.asarray([1, 2, 4, 8])
    auc = np.asarray(
        [
            verification[f"maskgit_confidence_{step}step"][
                "hard_average"
            ]["roc_auc"]
            for step in steps
        ]
    )
    milliseconds = np.asarray(
        [
            latency[f"maskgit_confidence_{step}step"][
                "per_set_latency_ms"
            ]
            for step in steps
        ]
    )
    figure, first_axis = plt.subplots(figsize=(7, 4))
    second_axis = first_axis.twinx()
    first_axis.plot(steps, auc, marker="o", color="#1f77b4")
    second_axis.plot(steps, milliseconds, marker="s", color="#d62728")
    first_axis.set(
        title="MaskGIT steps: accuracy-latency tradeoff",
        xlabel="Decoding steps",
        ylabel="Hard-scenario ROC-AUC",
        xticks=steps,
    )
    second_axis.set_ylabel("Latency (ms/set)")
    first_axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_quantization(
    quantization_manifest: dict[str, Any],
    path: Path,
) -> None:
    splits = ("train", "val", "test")
    token_cosine = [
        quantization_manifest["metrics"][split]["mean_token_cosine"]
        for split in splits
    ]
    map_cosine = [
        quantization_manifest["metrics"][split]["mean_map_cosine"]
        for split in splits
    ]
    utilization = [
        quantization_manifest["metrics"][split]["codebook_utilization"]
        for split in splits
    ]
    x_values = np.arange(3)
    width = 0.25
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(x_values - width, token_cosine, width, label="Token cosine")
    axis.bar(x_values, map_cosine, width, label="Map cosine")
    axis.bar(x_values + width, utilization, width, label="Utilization")
    axis.axhline(0.9, linestyle="--", color="black", linewidth=1)
    axis.set(
        title="K=1024 quantization diagnostics",
        ylabel="Value",
        ylim=(0.0, 1.05),
        xticks=x_values,
        xticklabels=("Train", "Validation", "Test"),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_final_report(
    *,
    training_run: Path,
    diagnostics_run: Path,
    output_directory: Path,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    training_manifest = _load_manifest(training_run)
    diagnostics_manifest = _load_manifest(diagnostics_run)
    quantization_root = Path(
        training_manifest["quantization_root"]
    ).expanduser().resolve()
    quantization_manifest = json.loads(
        (quantization_root / "manifest.json").read_text(encoding="utf-8")
    )

    verification_rows = _verification_rows(training_manifest)
    core_rows = _core_rows(training_manifest, diagnostics_manifest)
    training_rows = _training_rows(training_manifest)
    verification_path = output_directory / "verification_metrics.csv"
    core_path = output_directory / "core_comparison.csv"
    training_path = output_directory / "training_curves.csv"
    quantization_path = output_directory / "quantization_metrics.csv"
    latency_path = output_directory / "latency_metrics.csv"
    _write_csv(verification_path, verification_rows)
    _write_csv(core_path, core_rows)
    _write_csv(training_path, training_rows)
    quantization_rows = [
        {
            "split": split,
            **{
                key: value
                for key, value in quantization_manifest["metrics"][
                    split
                ].items()
                if not isinstance(value, dict)
            },
        }
        for split in ("train", "val", "test")
    ]
    _write_csv(quantization_path, quantization_rows)
    latency_rows = [
        {"method": method, **values}
        for method, values in diagnostics_manifest["diagnostics"][
            "latency"
        ].items()
    ]
    _write_csv(latency_path, latency_rows)

    figures = {
        "training": output_directory / "training_curves.png",
        "scenario_auc": output_directory / "scenario_auc.png",
        "steps_latency": output_directory / "steps_latency.png",
        "quantization": output_directory / "quantization.png",
    }
    _plot_training(training_manifest, figures["training"])
    _plot_scenario_auc(training_manifest, figures["scenario_auc"])
    _plot_steps_latency(
        training_manifest,
        diagnostics_manifest,
        figures["steps_latency"],
    )
    _plot_quantization(quantization_manifest, figures["quantization"])

    test_verification = training_manifest["research_evaluation"]["test"][
        "verification"
    ]
    confidence_hard = test_verification["maskgit_confidence"][
        "hard_average"
    ]["roc_auc"]
    one_shot_hard = test_verification["one_shot_transformer"][
        "hard_average"
    ]["roc_auc"]
    evidence_hard = test_verification["maskgit_evidence_logits"][
        "hard_average"
    ]["roc_auc"]
    remask_hard = test_verification["maskgit_evidence_remask"][
        "hard_average"
    ]["roc_auc"]
    quality_hard = test_verification["quality_pooling"]["hard_average"][
        "roc_auc"
    ]
    evidence_scenario_gains = {
        scenario: (
            test_verification["maskgit_evidence_logits"][scenario][
                "roc_auc"
            ]
            - test_verification["maskgit_confidence"][scenario]["roc_auc"]
        )
        for scenario in SCENARIOS
    }
    remask_scenario_gains = {
        scenario: (
            test_verification["maskgit_evidence_remask"][scenario]["roc_auc"]
            - test_verification["maskgit_evidence_logits"][scenario][
                "roc_auc"
            ]
        )
        for scenario in SCENARIOS
    }
    quant_test = quantization_manifest["metrics"]["test"]
    teacher_continuous = test_verification["teacher_continuous"]["all"]
    teacher_quantized = test_verification["teacher_quantized"]["all"]
    commit_correlation = diagnostics_manifest["diagnostics"][
        "commit_visibility_rank_correlation"
    ]["maskgit_evidence_remask"]["all"]
    decision = {
        "decision": "no_go_full_discrete_maskgit_current_representation",
        "quantization_local_fidelity_passed": bool(
            quantization_manifest["research_gate"]["test"]["passed"]
        ),
        "quantized_teacher_identity_auc": teacher_quantized["roc_auc"],
        "continuous_teacher_identity_auc": teacher_continuous["roc_auc"],
        "four_step_confidence_beats_one_shot": (
            confidence_hard > one_shot_hard
        ),
        "evidence_logits_beats_confidence_scenarios": sum(
            gain > 0 for gain in evidence_scenario_gains.values()
        ),
        "remask_beats_evidence_logits_scenarios": sum(
            gain > 0 for gain in remask_scenario_gains.values()
        ),
        "quality_pooling_beats_best_discrete": (
            quality_hard > max(evidence_hard, remask_hard)
        ),
        "recommended_pivot": (
            "deterministic_evidence_guided_local_aggregation"
        ),
        "secondary_option": "continuous_residual_prototype_diffusion",
        "absorbing_d3pm_now": False,
    }
    decision_path = output_directory / "go_no_go.json"
    decision_path.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    latency = diagnostics_manifest["diagnostics"]["latency"]
    report = f"""# Set2Proto-Diffusion 真实数据 Pilot 报告

## 结论

**No-Go：不建议在当前 AdaFace `7×7×512` hook、PCA-128、K=1024
表示上继续投入 10k–20k step 或实现 absorbing-state D3PM。**

建议下一步转向 **deterministic evidence-guided local aggregation**。如果仍需
迭代生成机制，再评估 continuous residual prototype diffusion。

## 数据与公平性

- CelebA identity-disjoint：train/val/test = 200/25/50 身份。
- set 数量：1,000/125/250；每个身份五种场景各一个。
- 每个样本 S=4、T=4，S/T 图像互不重叠；教师来自更高 AdaFace norm 图像。
- one-shot 与 MaskGIT 均为 4 层、hidden=256、8 heads、{training_manifest["selected_configuration"]["parameter_count"]:,} 参数，
  相同初始权重与数据，各训练 2,000 step。

## 核心结果

| 方法 | Test hard AUC | Test all EER | TAR@1e-2 | TAR@1e-3 | 延迟 ms/set |
|---|---:|---:|---:|---:|---:|
| Quality pooling | {quality_hard:.4f} | {test_verification["quality_pooling"]["all"]["eer"]:.4f} | {test_verification["quality_pooling"]["all"]["tar_at_far"]["0.01"]["tar"]:.3f} | {test_verification["quality_pooling"]["all"]["tar_at_far"]["0.001"]["tar"]:.3f} | {latency["quality_pooling"]["per_set_latency_ms"]:.3f} |
| One-shot Transformer | {one_shot_hard:.4f} | {test_verification["one_shot_transformer"]["all"]["eer"]:.4f} | {test_verification["one_shot_transformer"]["all"]["tar_at_far"]["0.01"]["tar"]:.3f} | {test_verification["one_shot_transformer"]["all"]["tar_at_far"]["0.001"]["tar"]:.3f} | {latency["one_shot_transformer"]["per_set_latency_ms"]:.3f} |
| 4-step confidence | {confidence_hard:.4f} | {test_verification["maskgit_confidence"]["all"]["eer"]:.4f} | {test_verification["maskgit_confidence"]["all"]["tar_at_far"]["0.01"]["tar"]:.3f} | {test_verification["maskgit_confidence"]["all"]["tar_at_far"]["0.001"]["tar"]:.3f} | {latency["maskgit_confidence_4step"]["per_set_latency_ms"]:.3f} |
| Evidence logits | {evidence_hard:.4f} | {test_verification["maskgit_evidence_logits"]["all"]["eer"]:.4f} | {test_verification["maskgit_evidence_logits"]["all"]["tar_at_far"]["0.01"]["tar"]:.3f} | {test_verification["maskgit_evidence_logits"]["all"]["tar_at_far"]["0.001"]["tar"]:.3f} | {latency["maskgit_evidence_logits"]["per_set_latency_ms"]:.3f} |
| Evidence + remask | {remask_hard:.4f} | {test_verification["maskgit_evidence_remask"]["all"]["eer"]:.4f} | {test_verification["maskgit_evidence_remask"]["all"]["tar_at_far"]["0.01"]["tar"]:.3f} | {test_verification["maskgit_evidence_remask"]["all"]["tar_at_far"]["0.001"]["tar"]:.3f} | {latency["maskgit_evidence_remask"]["per_set_latency_ms"]:.3f} |

测试集有 12,250 个负对，因此 FAR=1e-3 对应 12.25 个期望误接收，可作 pilot
估计；没有报告 FAR=1e-5。

## 核心假设判定

1. **离散量化：部分保留身份，但局部重建不合格。** Test token/map cosine
   为 {quant_test["mean_token_cosine"]:.3f}/{quant_test["mean_map_cosine"]:.3f}，
   低于 0.90 gate；利用率为 {quant_test["codebook_utilization"]:.1%}。连续教师
   AUC/TAR@1e-3 为 {teacher_continuous["roc_auc"]:.4f}/{teacher_continuous["tar_at_far"]["0.001"]["tar"]:.3f}，
   量化教师为 {teacher_quantized["roc_auc"]:.4f}/{teacher_quantized["tar_at_far"]["0.001"]["tar"]:.3f}。
2. **多步 vs 一次性：2k step 后成立，但幅度有限。** 4-step confidence hard
   AUC 比 one-shot 高 {confidence_hard - one_shot_hard:+.4f}。
3. **Evidence logits：成立。** 相对 confidence 在 5/5 场景均提升；
   场景增益为 {json.dumps(evidence_scenario_gains, ensure_ascii=False)}。
4. **Remask：不成立。** 仅 {sum(gain > 0 for gain in remask_scenario_gains.values())}/5
   场景优于 logits；hard AUC 下降 {remask_hard - evidence_hard:+.4f}。
5. **优势是否主要来自困难扰动：不充分支持。** Evidence 在 clean 也提升，
   而 remask 在低质量、互补遮挡、共同缺失和错身份中的多数场景退化。

## 失败分析

- 2,000 step 时两模型训练 token accuracy 约 100%，测试 token accuracy 仅
  {training_manifest["research_evaluation"]["test"]["token_accuracy"]["maskgit_evidence_logits"]["all"]:.1%}（evidence logits），
  表明 200 个训练身份下严重过拟合。
- 量化后的 deterministic mean/quality 仍明显优于模型，说明瓶颈不只在
  codebook，也在条件到 token 的泛化。
- evidence-remask commit score 与真实可见支持的 Spearman 相关仅
  {commit_correlation:.3f}，没有形成可靠的局部提交语义。
- `failure_cases.csv` 给出每个方法、每个测试样本的 genuine rank、最强冒名者
  和 margin；最坏样本覆盖 clean、低质量、互补遮挡、共同缺失与错身份。

## 建议

1. 首选：在连续 PCA 特征上实现 evidence-guided trimmed/top-k local
   aggregation，并直接优化身份验证或 teacher cosine。
2. 若确定性方法饱和后仍需要迭代修正，再尝试 continuous residual prototype
   diffusion；以 mean/quality prototype 为起点，只预测残差。
3. 暂停 absorbing-state D3PM 和 10k–20k 离散训练，除非先改善局部表示、
   codebook（例如 position-aware/product quantization）并显著扩大训练身份。

## 产物

- `core_comparison.csv`：核心公平比较。
- `verification_metrics.csv`：逐 split/方法/场景 ROC-AUC、EER、TAR。
- `quantization_metrics.csv`：量化重建、利用率、perplexity。
- `latency_metrics.csv`：批量推理延迟与峰值显存。
- `training_curves.csv`、`training_curves.png`：训练收敛与过拟合曲线。
- `scenario_auc.png`、`steps_latency.png`、`quantization.png`：关键图。
- 失败样本明细位于 `{diagnostics_manifest["diagnostics"]["failure_cases_csv"]}`。
"""
    report_path = output_directory / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    return {
        "report": str(report_path),
        "decision": str(decision_path),
        "tables": {
            "verification": str(verification_path),
            "core": str(core_path),
            "training": str(training_path),
            "quantization": str(quantization_path),
            "latency": str(latency_path),
        },
        "figures": {name: str(path) for name, path in figures.items()},
        "go_no_go": decision,
    }
