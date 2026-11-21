"""Colab GPU training script for Stage 1 Multi-Task Transformer classifier."""

import argparse
import json
from pathlib import Path

import pandas as pd

from draft.classifier.base import LabelEncoderPair
from draft.classifier.multitask import MultiTaskTrainer
from draft.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Colab multi-task classifier training."""
    parser = argparse.ArgumentParser(description="Train Stage 1 Multi-Task Email Classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Checkpoint output directory (defaults to <artifacts_dir>/classifier).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on training rows for quick smoke runs.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision during training.",
    )
    return parser.parse_args()


def _print_split_summary(split_name: str, metrics: dict) -> None:
    """Print concise accuracy, macro-F1, and weighted-F1 for a split."""
    intent = metrics["intent"]
    urgency = metrics["urgency"]
    print(
        f"  [{split_name}] "
        f"intent(acc={intent['accuracy']:.4f}, macro_f1={intent['macro_f1']:.4f}, "
        f"weighted_f1={intent['weighted_f1']:.4f}) | "
        f"urgency(acc={urgency['accuracy']:.4f}, macro_f1={urgency['macro_f1']:.4f}, "
        f"weighted_f1={urgency['weighted_f1']:.4f}) | "
        f"combined_macro_f1={metrics['combined_macro_f1']:.4f}"
    )


def main() -> None:
    """Load canonical splits, train MultiTaskTrainer, and save reports."""
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.classifier.epochs = args.epochs
    if args.batch_size is not None:
        cfg.classifier.batch_size = args.batch_size

    processed_dir = cfg.paths.processed_dir
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    if args.max_train_samples is not None and args.max_train_samples < len(train_df):
        train_df = train_df.sample(
            n=args.max_train_samples,
            random_state=cfg.data.random_seed,
        ).reset_index(drop=True)

    output_dir = args.output_dir or (cfg.paths.artifacts_dir / "classifier")
    labels = LabelEncoderPair.from_taxonomy(cfg.taxonomy)
    trainer = MultiTaskTrainer(config=cfg, labels=labels, use_amp=not args.no_amp)
    summary = trainer.fit(train_df=train_df, val_df=val_df, test_df=test_df, output_dir=output_dir)

    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.paths.reports_dir / "classifier_transformer_metrics.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Training finished (backbone={cfg.classifier.backbone_model}, "
        f"best_epoch={summary['best_epoch']}/{summary['epochs']}):"
    )
    _print_split_summary("val", summary["val"])
    _print_split_summary("test", summary["test"])
    print(f"  Checkpoint directory (copy locally): {output_dir}")
    print(f"  Metrics report: {report_path}")


if __name__ == "__main__":
    main()
