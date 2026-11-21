"""Colab GPU training script for Stage 2 contrastive bi-encoder fine-tuning."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from draft.config import AppConfig, load_config
from draft.data.org_docs import load_org_docs
from draft.retrieval.dual_retriever import DualSourceRetriever
from draft.retrieval.encoders import SentenceTransformerEncoder
from draft.retrieval.hybrid_store import HybridVectorStore
from draft.retrieval.metrics import evaluate_org_doc_retrieval, evaluate_reference_retrieval
from draft.retrieval.trainer import BiEncoderFineTuner
from draft.schemas import OrgDocument


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Colab bi-encoder fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune Stage 2 Bi-Encoder Retriever.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--eval-queries", type=int, default=300)
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


def _evaluate_encoder(
    cfg: AppConfig,
    encoder: SentenceTransformerEncoder,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    org_docs: list[OrgDocument],
    max_queries: int,
) -> dict[str, Any]:
    """Build in-memory stores and evaluate encoder on validation retrieval."""
    doc_texts = [f"{doc.title}\n\n{doc.content}" for doc in org_docs]
    doc_meta = [doc.model_dump(mode="json") for doc in org_docs]
    doc_store = HybridVectorStore.build(doc_texts, encoder.encode(doc_texts), doc_meta)

    ref_texts = train_df["email"].astype(str).tolist()
    ref_meta = [
        {
            "email_id": str(row["email_id"]),
            "email": str(row["email"]),
            "answer": str(row["answer"]),
            "intent": str(row["intent"]),
            "urgency": str(row["urgency"]),
            "tags": [str(t) for t in row["tags"]] if row.get("tags") is not None else [],
        }
        for row in train_df.to_dict(orient="records")
    ]
    ref_store = HybridVectorStore.build(ref_texts, encoder.encode(ref_texts), ref_meta)

    retriever = DualSourceRetriever(cfg, encoder, doc_store, ref_store)
    return {
        "org_docs": evaluate_org_doc_retrieval(
            retriever, val_df, max_queries=max_queries, seed=cfg.data.random_seed
        ),
        "ref_emails": evaluate_reference_retrieval(
            retriever, val_df, max_queries=max_queries, seed=cfg.data.random_seed
        ),
    }


def main() -> None:
    """Evaluate pretrained bi-encoder, fine-tune on train pairs, and re-evaluate."""
    args = parse_args()
    cfg = load_config(args.config)
    train_df = pd.read_parquet(cfg.paths.processed_dir / "train.parquet")
    val_df = pd.read_parquet(cfg.paths.processed_dir / "val.parquet")
    org_docs = load_org_docs(cfg.paths.org_docs_file)

    output_dir = args.output_dir or (cfg.paths.artifacts_dir / "retriever" / "encoder")
    base_encoder = SentenceTransformerEncoder(cfg.retrieval.embedding_model, show_progress_bar=True)
    before = _evaluate_encoder(cfg, base_encoder, train_df, val_df, org_docs, args.eval_queries)

    BiEncoderFineTuner(cfg.retrieval.embedding_model).fit(
        train_df=train_df,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_train_samples,
        seed=cfg.data.random_seed,
    )

    tuned_encoder = SentenceTransformerEncoder(output_dir, show_progress_bar=True)
    after = _evaluate_encoder(cfg, tuned_encoder, train_df, val_df, org_docs, args.eval_queries)

    if args.rebuild_index:
        DualSourceRetriever.build_and_save_indices(cfg, tuned_encoder, train_df, org_docs)

    summary = {"base_model": cfg.retrieval.embedding_model, "before": before, "after": after}
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.paths.reports_dir / "retriever_training_metrics.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Bi-encoder fine-tuning completed:")
    print(f"  Before ref intent_mrr@3={before['ref_emails']['intent_mrr@3']:.4f}")
    print(f"  After  ref intent_mrr@3={after['ref_emails']['intent_mrr@3']:.4f}")
    print(f"  Encoder saved to: {output_dir}")
    print(f"  Metrics saved to: {report_path}")


if __name__ == "__main__":
    main()
