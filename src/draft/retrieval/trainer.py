"""Contrastive bi-encoder fine-tuning utilities for Stage 2 retrieval."""

from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer

try:
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.training_args import (
        BatchSamplers,
        SentenceTransformerTrainingArguments,
    )
except ImportError:  # pragma: no cover - fallback for older sentence-transformers releases
    from sentence_transformers import SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers


def build_contrastive_pairs(
    train_df: pd.DataFrame,
    max_samples: int | None = None,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Build positive (email, answer) pairs exclusively from the training split."""
    working = train_df
    if max_samples is not None and max_samples < len(train_df):
        working = train_df.sample(n=max_samples, random_state=seed)
    anchors = working["email"].astype(str).tolist()
    positives = working["answer"].astype(str).tolist()
    return anchors, positives


class BiEncoderFineTuner:
    """Contrastive bi-encoder trainer using MultipleNegativesRankingLoss."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device

    def fit(
        self,
        train_df: pd.DataFrame,
        output_dir: Path,
        epochs: int = 2,
        batch_size: int = 64,
        lr: float = 2e-5,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> Path:
        """Fine-tune bi-encoder on (email, answer) pairs and save the resulting model."""
        anchors, positives = build_contrastive_pairs(train_df, max_samples=max_samples, seed=seed)
        train_dataset = Dataset.from_dict({"anchor": anchors, "positive": positives})

        model = SentenceTransformer(self.model_name, device=self.device)
        loss = MultipleNegativesRankingLoss(model)

        steps_per_epoch = max(len(anchors) // batch_size, 1)
        warmup_steps = max(1, int(0.1 * steps_per_epoch * epochs))
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and not use_bf16

        args = SentenceTransformerTrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            warmup_steps=warmup_steps,
            fp16=use_fp16,
            bf16=use_bf16,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            save_strategy="no",
            logging_steps=50,
            report_to="none",
            seed=seed,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            loss=loss,
        )
        trainer.train()

        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        return output_dir
