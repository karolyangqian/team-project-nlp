"""Pipeline evaluator running end-to-end ticket, policy, and ablation benchmarks."""

from typing import Any

import pandas as pd

from draft.classifier.base import LabelEncoderPair, compute_classification_metrics
from draft.evaluation.metrics import (
    build_policy_eval_suite,
    compute_bertscore,
    compute_placeholder_leakage_rate,
    compute_policy_adherence,
    compute_rouge_l,
)
from draft.generation.prompts import normalize_target_reply
from draft.pipeline import DraftPipeline
from draft.schemas import DraftResponse, IncomingEmail, OrgDocument


class PipelineEvaluator:
    """Evaluates classification, retrieval, generation quality, policy adherence, and latency."""

    def __init__(self, pipeline: DraftPipeline) -> None:
        self.pipeline = pipeline
        self.labels = LabelEncoderPair.from_taxonomy(pipeline.config.taxonomy)

    @staticmethod
    def _latency_stats(timings: list[dict[str, float]]) -> dict[str, dict[str, float]]:
        """Compute mean and p90 milliseconds for each pipeline stage."""
        if not timings:
            return {}
        stats: dict[str, dict[str, float]] = {}
        for key in ("stage1_ms", "stage2_ms", "stage3_ms", "total_ms"):
            series = pd.Series([item[key] for item in timings])
            stats[key] = {"mean": float(series.mean()), "p90": float(series.quantile(0.90))}
        return stats

    def _classification_summary(
        self,
        rows: list[dict[str, Any]],
        drafts: list[DraftResponse],
    ) -> dict[str, float]:
        """Compute Stage 1 accuracy and macro-F1 on evaluated ticket records."""
        metrics = compute_classification_metrics(
            y_true_intent=[self.labels.encode_intent(r["intent"]) for r in rows],
            y_pred_intent=[self.labels.encode_intent(d.classification.intent) for d in drafts],
            y_true_urgency=[self.labels.encode_urgency(r["urgency"]) for r in rows],
            y_pred_urgency=[self.labels.encode_urgency(d.classification.urgency) for d in drafts],
            labels=self.labels,
        )
        return {
            "intent_accuracy": float(metrics["intent"]["accuracy"]),
            "intent_macro_f1": float(metrics["intent"]["macro_f1"]),
            "urgency_accuracy": float(metrics["urgency"]["accuracy"]),
            "urgency_macro_f1": float(metrics["urgency"]["macro_f1"]),
            "combined_macro_f1": float(metrics["combined_macro_f1"]),
        }

    def evaluate_tickets(
        self,
        eval_df: pd.DataFrame,
        max_samples: int = 50,
        use_rag: bool = True,
        use_reranker: bool = True,
        use_few_shot: bool = True,
        include_bertscore: bool = False,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Evaluate end-to-end draft quality on a stratified/random ticket sample."""
        sampled = (
            eval_df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
            if max_samples < len(eval_df)
            else eval_df.reset_index(drop=True)
        )
        rows = sampled.to_dict(orient="records")
        drafts: list[DraftResponse] = []
        timings: list[dict[str, float]] = []
        references: list[str] = []

        for row in rows:
            email = IncomingEmail(
                subject=str(row.get("subject", "")),
                body=str(row.get("body", row["email"])),
            )
            draft, timing = self.pipeline.run_with_timing(
                email=email,
                use_rag=use_rag,
                use_reranker=use_reranker,
                use_few_shot=use_few_shot,
                exclude_email_ids={str(row["email_id"])},
            )
            drafts.append(draft)
            timings.append(timing)
            references.append(normalize_target_reply(str(row["answer"])))

        predictions = [draft.body for draft in drafts]
        result: dict[str, Any] = {
            "num_samples": len(rows),
            "use_rag": use_rag,
            "use_reranker": use_reranker,
            "use_few_shot": use_few_shot,
            "model_name": self.pipeline.generator.model_name,
            "classification": self._classification_summary(rows, drafts),
            "rouge": compute_rouge_l(predictions, references),
            "policy_adherence": compute_policy_adherence(drafts),
            "latency_ms": self._latency_stats(timings),
            "samples": [
                {
                    "email_id": str(row["email_id"]),
                    "true_intent": str(row["intent"]),
                    "pred_intent": str(draft.classification.intent),
                    "true_urgency": str(row["urgency"]),
                    "pred_urgency": str(draft.classification.urgency),
                    "retrieved_doc_ids": [d.doc_id for d in draft.retrieved_context.top_k_docs],
                    "cited_doc_ids": draft.cited_doc_ids,
                    "reference_answer": ref,
                    "draft_body": draft.body,
                }
                for row, draft, ref in zip(rows[:5], drafts[:5], references[:5], strict=True)
            ],
        }
        if include_bertscore:
            result["bertscore"] = compute_bertscore(predictions, references)
        return result

    def evaluate_policy_suite(
        self,
        org_docs: list[OrgDocument],
        use_rag: bool = True,
        use_reranker: bool = True,
        use_few_shot: bool = True,
    ) -> dict[str, Any]:
        """Evaluate factual adherence across 12 unseen policy test scenarios."""
        suite = build_policy_eval_suite(org_docs)
        if not suite:
            return {
                "num_scenarios": 0,
                "doc_retrieval_hit_rate": 0.0,
                "target_citation_rate": 0.0,
                "policy_fact_recall": 0.0,
                "placeholder_leakage_rate": 0.0,
                "scenarios": [],
            }

        retrieval_hits = 0
        citation_hits = 0
        fact_recalls: list[float] = []
        draft_bodies: list[str] = []
        details: list[dict[str, Any]] = []

        for item in suite:
            email = IncomingEmail(subject=item["subject"], body=item["body"])
            draft, _ = self.pipeline.run_with_timing(
                email=email,
                use_rag=use_rag,
                use_reranker=use_reranker,
                use_few_shot=use_few_shot,
            )
            draft_bodies.append(draft.body)
            retrieved_ids = [d.doc_id for d in draft.retrieved_context.top_k_docs]
            doc_hit = item["expected_doc_id"] in retrieved_ids
            cite_hit = item["expected_doc_id"] in draft.cited_doc_ids
            body_lower = draft.body.lower()
            matched_facts = [
                fact for fact in item["expected_facts"] if fact.lower() in body_lower
            ]
            recall = len(matched_facts) / float(max(len(item["expected_facts"]), 1))

            retrieval_hits += int(doc_hit)
            citation_hits += int(cite_hit)
            fact_recalls.append(recall)
            details.append(
                {
                    "scenario_id": item["scenario_id"],
                    "expected_doc_id": item["expected_doc_id"],
                    "retrieved_doc_ids": retrieved_ids,
                    "cited_doc_ids": draft.cited_doc_ids,
                    "doc_retrieval_hit": doc_hit,
                    "target_citation_hit": cite_hit,
                    "fact_recall": recall,
                    "matched_facts": matched_facts,
                    "draft_body": draft.body,
                }
            )

        n = float(len(suite))
        return {
            "num_scenarios": len(suite),
            "use_rag": use_rag,
            "use_few_shot": use_few_shot,
            "doc_retrieval_hit_rate": retrieval_hits / n,
            "target_citation_rate": citation_hits / n,
            "policy_fact_recall": sum(fact_recalls) / n,
            "placeholder_leakage_rate": compute_placeholder_leakage_rate(draft_bodies),
            "scenarios": details,
        }

    def run_ablation(
        self,
        eval_df: pd.DataFrame,
        org_docs: list[OrgDocument],
        max_samples: int = 50,
        use_reranker: bool = True,
        use_few_shot: bool = True,
        include_bertscore: bool = False,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Compare no_rag vs with_rag modes across tickets and policy adherence suite."""
        results: dict[str, Any] = {}
        for mode_name, rag_enabled in (("no_rag", False), ("with_rag", True)):
            results[mode_name] = {
                "tickets": self.evaluate_tickets(
                    eval_df=eval_df,
                    max_samples=max_samples,
                    use_rag=rag_enabled,
                    use_reranker=use_reranker,
                    use_few_shot=use_few_shot,
                    include_bertscore=include_bertscore,
                    seed=seed,
                ),
                "policy_suite": self.evaluate_policy_suite(
                    org_docs=org_docs,
                    use_rag=rag_enabled,
                    use_reranker=use_reranker,
                    use_few_shot=use_few_shot,
                ),
            }
        return results
