"""Run comprehensive Stage 1, Stage 2, and Stage 3 experiments for presentation reports."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from draft.classifier.base import LabelEncoderPair, compute_classification_metrics
from draft.config import AppConfig, load_config
from draft.data.org_docs import load_org_docs
from draft.evaluation.evaluator import PipelineEvaluator
from draft.evaluation.metrics import (
    build_policy_eval_suite,
    compute_bertscore,
    compute_placeholder_leakage_rate,
    compute_policy_adherence,
    compute_rouge_l,
)
from draft.generation.prompts import normalize_target_reply
from draft.pipeline import DraftPipeline
from draft.retrieval.dual_retriever import DualSourceRetriever
from draft.schemas import DraftResponse, IncomingEmail


def _stratified_ticket_sample(
    df: pd.DataFrame,
    per_intent: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample balanced test tickets across every intent class."""
    parts = []
    for _, group in df.groupby("intent", sort=True):
        n = min(per_intent, len(group))
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def _top_confusions(
    confusion_matrix: list[list[int]],
    labels: list[str],
    top_n: int = 8,
) -> list[dict[str, Any]]:
    """Extract highest off-diagonal confusion pairs from a confusion matrix."""
    pairs = []
    for i, true_label in enumerate(labels):
        row_total = sum(confusion_matrix[i])
        for j, pred_label in enumerate(confusion_matrix[i]):
            if i != j and pred_label > 0:
                pairs.append(
                    {
                        "true": true_label,
                        "pred": pred_label_name(labels, j),
                        "count": int(pred_label),
                        "row_share": float(pred_label / max(row_total, 1)),
                    }
                )
    pairs.sort(key=lambda item: item["count"], reverse=True)
    return pairs[:top_n]


def pred_label_name(labels: list[str], idx: int) -> str:
    """Return label name safely by index."""
    return labels[idx]


def run_stage1_analysis(cfg: AppConfig, test_df: pd.DataFrame) -> dict[str, Any]:
    """Benchmark Stage 1 latency and extract confusion errors on the test split."""
    import time

    baseline_metrics = json.loads(
        (cfg.paths.reports_dir / "classifier_baseline_metrics.json").read_text(encoding="utf-8")
    )
    transformer_metrics = json.loads(
        (cfg.paths.reports_dir / "classifier_transformer_metrics.json").read_text(encoding="utf-8")
    )
    labels = LabelEncoderPair.from_taxonomy(cfg.taxonomy)

    sample_rows = _stratified_ticket_sample(test_df, per_intent=5, seed=cfg.data.random_seed)
    sample_emails = [
        IncomingEmail(subject=str(r.get("subject", "")), body=str(r.get("body", r["email"])))
        for r in sample_rows.to_dict(orient="records")
    ]

    latencies: dict[str, dict[str, float]] = {}
    sample_errors: dict[str, list[dict[str, Any]]] = {}

    for backend in ("baseline", "transformer"):
        clf = DraftPipeline.load_classifier(cfg, backend=backend)
        _ = clf.predict(sample_emails[0])
        times = []
        preds = []
        for em in sample_emails:
            t0 = time.perf_counter()
            pred = clf.predict(em)
            times.append((time.perf_counter() - t0) * 1000.0)
            preds.append(pred)
        series = pd.Series(times)
        latencies[backend] = {
            "mean_ms": float(series.mean()),
            "p90_ms": float(series.quantile(0.90)),
        }
        errors = []
        for row, pred in zip(sample_rows.to_dict(orient="records"), preds, strict=True):
            if pred.intent != row["intent"] or pred.urgency != row["urgency"]:
                errors.append(
                    {
                        "email_id": str(row["email_id"]),
                        "subject": str(row.get("subject", "")),
                        "body_preview": str(row.get("body", ""))[:180],
                        "true_intent": str(row["intent"]),
                        "pred_intent": str(pred.intent),
                        "intent_conf": round(pred.intent_confidence, 4),
                        "true_urgency": str(row["urgency"]),
                        "pred_urgency": str(pred.urgency),
                    }
                )
        sample_errors[backend] = errors[:8]

    intent_names = [str(x) for x in labels.intent_labels]
    urgency_names = [str(x) for x in labels.urgency_labels]
    analysis = {
        "latency_ms": latencies,
        "top_intent_confusions": {
            "baseline": _top_confusions(
                baseline_metrics["test"]["intent"]["confusion_matrix"], intent_names
            ),
            "transformer": _top_confusions(
                transformer_metrics["test"]["intent"]["confusion_matrix"], intent_names
            ),
        },
        "top_urgency_confusions": {
            "baseline": _top_confusions(
                baseline_metrics["test"]["urgency"]["confusion_matrix"], urgency_names, top_n=6
            ),
            "transformer": _top_confusions(
                transformer_metrics["test"]["urgency"]["confusion_matrix"],
                urgency_names,
                top_n=6,
            ),
        },
        "sample_errors": sample_errors,
    }
    out_path = cfg.paths.reports_dir / "stage1_analysis.json"
    out_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"[stage1] saved {out_path}")
    return analysis


def run_stage2_analysis(cfg: AppConfig, org_docs: list[Any]) -> dict[str, Any]:
    """Compare reranker ON vs OFF exact policy retrieval across all 12 unseen scenarios."""
    suite = build_policy_eval_suite(org_docs)
    retriever = DualSourceRetriever.load(cfg, enable_reranker=True)

    runs: dict[str, Any] = {}
    for clf_name in ("baseline", "transformer"):
        clf = DraftPipeline.load_classifier(cfg, backend=clf_name)
        for rerank in (False, True):
            key = f"{clf_name}_rerank_{'on' if rerank else 'off'}"
            top1_hits = 0
            top3_hits = 0
            cat_top1_hits = 0
            scenarios = []
            for item in suite:
                email = IncomingEmail(subject=item["subject"], body=item["body"])
                cls = clf.predict(email)
                ctx = retriever.retrieve(email, intent=cls.intent, use_reranker=rerank)
                doc_ids = [d.doc_id for d in ctx.top_k_docs]
                doc_cats = [str(d.category) for d in ctx.top_k_docs]
                hit1 = bool(doc_ids) and doc_ids[0] == item["expected_doc_id"]
                hit3 = item["expected_doc_id"] in doc_ids
                cat1 = bool(doc_cats) and doc_cats[0] == item["expected_category"]
                top1_hits += int(hit1)
                top3_hits += int(hit3)
                cat_top1_hits += int(cat1)
                scenarios.append(
                    {
                        "scenario_id": item["scenario_id"],
                        "expected_doc_id": item["expected_doc_id"],
                        "expected_category": item["expected_category"],
                        "pred_intent": str(cls.intent),
                        "retrieved_doc_ids": doc_ids,
                        "top1_hit": hit1,
                        "top3_hit": hit3,
                    }
                )
            n = float(len(suite))
            runs[key] = {
                "classifier": clf_name,
                "reranker": rerank,
                "exact_doc_hit@1": top1_hits / n,
                "exact_doc_hit@3": top3_hits / n,
                "exact_category_hit@1": cat_top1_hits / n,
                "scenarios": scenarios,
            }

    out_path = cfg.paths.reports_dir / "stage2_analysis.json"
    out_path.write_text(json.dumps({"policy_suite_retrieval": runs}, indent=2), encoding="utf-8")
    print(f"[stage2] saved {out_path}")
    return runs


def _evaluate_custom_tickets(
    pipeline: DraftPipeline,
    sampled_df: pd.DataFrame,
    use_rag: bool,
    use_reranker: bool,
    use_few_shot: bool,
    include_bertscore: bool,
) -> dict[str, Any]:
    """Evaluate a DraftPipeline on a fixed stratified ticket dataframe."""
    labels = LabelEncoderPair.from_taxonomy(pipeline.config.taxonomy)
    rows = sampled_df.to_dict(orient="records")
    drafts: list[DraftResponse] = []
    timings: list[dict[str, float]] = []
    references: list[str] = []

    for idx, row in enumerate(rows, start=1):
        email = IncomingEmail(
            subject=str(row.get("subject", "")),
            body=str(row.get("body", row["email"])),
        )
        draft, timing = pipeline.run_with_timing(
            email=email,
            use_rag=use_rag,
            use_reranker=use_reranker,
            use_few_shot=use_few_shot,
            exclude_email_ids={str(row["email_id"])},
        )
        drafts.append(draft)
        timings.append(timing)
        references.append(normalize_target_reply(str(row["answer"])))
        print(
            f"    ticket {idx}/{len(rows)} ({row['intent']}): "
            f"s3={timing['stage3_ms']:.0f}ms cited={draft.cited_doc_ids}"
        )

    predictions = [d.body for d in drafts]
    cls_metrics = compute_classification_metrics(
        y_true_intent=[labels.encode_intent(r["intent"]) for r in rows],
        y_pred_intent=[labels.encode_intent(d.classification.intent) for d in drafts],
        y_true_urgency=[labels.encode_urgency(r["urgency"]) for r in rows],
        y_pred_urgency=[labels.encode_urgency(d.classification.urgency) for d in drafts],
        labels=labels,
    )
    latency_stats: dict[str, dict[str, float]] = {}
    for key in ("stage1_ms", "stage2_ms", "stage3_ms", "total_ms"):
        series = pd.Series([item[key] for item in timings])
        latency_stats[key] = {"mean": float(series.mean()), "p90": float(series.quantile(0.90))}

    result: dict[str, Any] = {
        "num_samples": len(rows),
        "use_rag": use_rag,
        "use_reranker": use_reranker,
        "use_few_shot": use_few_shot,
        "model_name": pipeline.generator.model_name,
        "classification": {
            "intent_accuracy": float(cls_metrics["intent"]["accuracy"]),
            "intent_macro_f1": float(cls_metrics["intent"]["macro_f1"]),
            "urgency_accuracy": float(cls_metrics["urgency"]["accuracy"]),
            "urgency_macro_f1": float(cls_metrics["urgency"]["macro_f1"]),
            "combined_macro_f1": float(cls_metrics["combined_macro_f1"]),
        },
        "rouge": compute_rouge_l(predictions, references),
        "policy_adherence": compute_policy_adherence(drafts),
        "placeholder_leakage_rate": compute_placeholder_leakage_rate(predictions),
        "latency_ms": latency_stats,
        "samples": [
            {
                "email_id": str(row["email_id"]),
                "subject": str(row.get("subject", "")),
                "body": str(row.get("body", "")),
                "true_intent": str(row["intent"]),
                "pred_intent": str(draft.classification.intent),
                "true_urgency": str(row["urgency"]),
                "pred_urgency": str(draft.classification.urgency),
                "retrieved_doc_ids": [d.doc_id for d in draft.retrieved_context.top_k_docs],
                "cited_doc_ids": draft.cited_doc_ids,
                "reference_answer": ref,
                "draft_body": draft.body,
                "stage3_ms": round(timing["stage3_ms"], 1),
            }
            for row, draft, ref, timing in zip(
                rows, drafts, references, timings, strict=True
            )
        ],
    }
    if include_bertscore:
        result["bertscore"] = compute_bertscore(predictions, references)
    return result


def run_stage3_ablation(
    cfg: AppConfig,
    test_df: pd.DataFrame,
    org_docs: list[Any],
    per_intent: int = 1,
    include_bertscore: bool = True,
) -> dict[str, Any]:
    """Run 5-way Stage 3 ablation on stratified test tickets and 12 policy scenarios."""
    out_path = cfg.paths.reports_dir / "stage3_ablation.json"
    existing: dict[str, Any] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))

    sampled_df = _stratified_ticket_sample(
        test_df, per_intent=per_intent, seed=cfg.data.random_seed
    )
    classifier = DraftPipeline.load_classifier(cfg, backend="baseline")
    retriever = DualSourceRetriever.load(cfg, enable_reranker=True)
    prompt_builder = DraftPipeline.from_config(
        cfg, classifier_backend="baseline", generator_backend="mock", enable_reranker=False
    ).prompt_builder

    modes = [
        ("qwen25_no_rag", "qwen2.5:3b", "Base Qwen2.5-3B (No RAG)", False, False),
        ("qwen25_rag_0shot", "qwen2.5:3b", "Base Qwen2.5-3B + RAG (0-Shot)", True, False),
        ("qwen25_rag_2shot", "qwen2.5:3b", "Base Qwen2.5-3B + RAG + 2-Shot", True, True),
        ("sft_rag_0shot", "draft-qwen25-3b", "LoRA SFT + RAG (0-Shot)", True, False),
        ("sft_rag_2shot", "draft-qwen25-3b", "LoRA SFT + RAG + 2-Shot", True, True),
    ]

    report: dict[str, Any] = {
        "split": "test",
        "stratified_per_intent": per_intent,
        "num_ticket_samples": len(sampled_df),
        "num_policy_scenarios": 12,
        "classifier_backend": "baseline",
        "reranker": True,
        "modes": existing.get("modes", {}),
    }

    generators: dict[str, Any] = {}
    for mode_key, model_tag, label, use_rag, use_few_shot in modes:
        if mode_key in report["modes"]:
            print(f"[stage3] skipping completed mode: {mode_key}")
            continue
        print(f"[stage3] running mode={mode_key} ({label})...")
        if model_tag not in generators:
            generators[model_tag] = DraftPipeline.load_generator(
                cfg, backend="ollama", ollama_model=model_tag
            )
        pipeline = DraftPipeline(cfg, classifier, retriever, prompt_builder, generators[model_tag])
        evaluator = PipelineEvaluator(pipeline)
        ticket_res = _evaluate_custom_tickets(
            pipeline=pipeline,
            sampled_df=sampled_df,
            use_rag=use_rag,
            use_reranker=True,
            use_few_shot=use_few_shot,
            include_bertscore=include_bertscore,
        )
        policy_res = evaluator.evaluate_policy_suite(
            org_docs=org_docs,
            use_rag=use_rag,
            use_reranker=True,
            use_few_shot=use_few_shot,
        )
        report["modes"][mode_key] = {
            "label": label,
            "ollama_model": model_tag,
            "use_rag": use_rag,
            "use_few_shot": use_few_shot,
            "tickets": ticket_res,
            "policy_suite": policy_res,
        }
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"  -> saved checkpoint: rougeL={ticket_res['rouge']['rougeL_f1']:.4f} "
            f"leak={ticket_res['placeholder_leakage_rate']:.2%} "
            f"policy_recall={policy_res['policy_fact_recall']:.4f} "
            f"cite={policy_res['target_citation_rate']:.4f}"
        )

    return report


def parse_args() -> argparse.Namespace:
    """Parse CLI options for presentation experiment suite."""
    parser = argparse.ArgumentParser(
        description="Run presentation experiments across all 3 stages."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--per-intent", type=int, default=1)
    parser.add_argument("--skip-stage3", action="store_true")
    parser.add_argument("--no-bertscore", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Execute Stage 1, Stage 2, and Stage 3 evaluation suites."""
    args = parse_args()
    cfg = load_config(args.config)
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    test_df = pd.read_parquet(cfg.paths.processed_dir / "test.parquet")
    org_docs = load_org_docs(cfg.paths.org_docs_file)

    run_stage1_analysis(cfg, test_df)
    run_stage2_analysis(cfg, org_docs)

    if not args.skip_stage3:
        run_stage3_ablation(
            cfg=cfg,
            test_df=test_df,
            org_docs=org_docs,
            per_intent=args.per_intent,
            include_bertscore=not args.no_bertscore,
        )

    # Also store intent/placeholder summary check for consistency
    intent_counts = Counter(test_df["intent"].tolist())
    print(f"Done. Stratified intents covered: {len(intent_counts)}")


if __name__ == "__main__":
    main()
