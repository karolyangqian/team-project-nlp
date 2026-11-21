"""Colab GPU 3-way ablation script comparing zero-shot, zero-shot+RAG, and fine-tuned+RAG."""

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from draft.config import load_config
from draft.data.org_docs import load_org_docs
from draft.evaluation.evaluator import PipelineEvaluator
from draft.generation.hf_backend import HFPeftGenerator
from draft.generation.prompts import PromptBuilder
from draft.pipeline import DraftPipeline
from draft.retrieval.dual_retriever import DualSourceRetriever


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Colab 3-way generation ablation."""
    parser = argparse.ArgumentParser(description="Run 3-way DRAFT generation ablation on GPU.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--classifier-backend", default="auto")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bertscore", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def _print_ablation_row(name: str, mode_result: dict[str, Any]) -> None:
    """Print one concise summary row for an evaluated ablation mode."""
    tickets = mode_result["tickets"]
    policy = mode_result["policy_suite"]
    rouge_l = tickets["rouge"]["rougeL_f1"]
    bscore = tickets.get("bertscore", {}).get("f1", 0.0)
    cite_rate = tickets["policy_adherence"]["citation_rate"]
    fact_rec = policy["policy_fact_recall"]
    total_ms = tickets["latency_ms"]["total_ms"]["mean"]
    print(
        f"  {name:20s} ROUGE-L={rouge_l:.4f} BERTScore={bscore:.4f} "
        f"CiteRate={cite_rate:.4f} PolicyFactRecall={fact_rec:.4f} MeanMs={total_ms:.1f}"
    )


def main() -> None:
    """Evaluate base_no_rag, base_with_rag, and finetuned_with_rag and save JSON report."""
    args = parse_args()
    cfg = load_config(args.config)
    eval_df = pd.read_parquet(cfg.paths.processed_dir / f"{args.split}.parquet")
    org_docs = load_org_docs(cfg.paths.org_docs_file)

    classifier = DraftPipeline.load_classifier(cfg, backend=args.classifier_backend)
    retriever = DualSourceRetriever.load(cfg, enable_reranker=args.rerank)
    prompt_builder = PromptBuilder(cfg.paths.prompts_dir)
    use_4bit = not args.no_4bit

    results: dict[str, Any] = {"split": args.split, "max_samples": args.max_samples, "modes": {}}

    base_gen = HFPeftGenerator(
        base_model_name=cfg.generation.base_model,
        adapter_dir=None,
        load_in_4bit=use_4bit,
        default_max_new_tokens=cfg.generation.max_new_tokens,
        default_temperature=cfg.generation.temperature,
    )
    base_pipeline = DraftPipeline(cfg, classifier, retriever, prompt_builder, base_gen)
    base_eval = PipelineEvaluator(base_pipeline)

    for mode_name, use_rag in (("base_no_rag", False), ("base_with_rag", True)):
        results["modes"][mode_name] = {
            "tickets": base_eval.evaluate_tickets(
                eval_df=eval_df,
                max_samples=args.max_samples,
                use_rag=use_rag,
                use_reranker=args.rerank,
                include_bertscore=args.bertscore,
                seed=cfg.data.random_seed,
            ),
            "policy_suite": base_eval.evaluate_policy_suite(
                org_docs=org_docs,
                use_rag=use_rag,
                use_reranker=args.rerank,
            ),
        }

    del base_gen, base_pipeline, base_eval
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    adapter_dir = args.adapter_dir or (cfg.paths.artifacts_dir / "llm" / "lora_adapter")
    ft_gen = HFPeftGenerator(
        base_model_name=cfg.generation.base_model,
        adapter_dir=adapter_dir,
        load_in_4bit=use_4bit,
        default_max_new_tokens=cfg.generation.max_new_tokens,
        default_temperature=cfg.generation.temperature,
    )
    ft_pipeline = DraftPipeline(cfg, classifier, retriever, prompt_builder, ft_gen)
    ft_eval = PipelineEvaluator(ft_pipeline)
    results["modes"]["finetuned_with_rag"] = {
        "tickets": ft_eval.evaluate_tickets(
            eval_df=eval_df,
            max_samples=args.max_samples,
            use_rag=True,
            use_reranker=args.rerank,
            include_bertscore=args.bertscore,
            seed=cfg.data.random_seed,
        ),
        "policy_suite": ft_eval.evaluate_policy_suite(
            org_docs=org_docs,
            use_rag=True,
            use_reranker=args.rerank,
        ),
    }

    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.paths.reports_dir / "ablation_table.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"3-way ablation results on {args.split} ({args.max_samples} samples):")
    for mode_name, mode_data in results["modes"].items():
        _print_ablation_row(mode_name, mode_data)
    print(f"  [report] {report_path}")


if __name__ == "__main__":
    main()
