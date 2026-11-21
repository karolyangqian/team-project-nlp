"""Multi-task transformer classifier, trainer, and local inference wrapper."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from safetensors.torch import load_model, save_model
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from draft.classifier.base import (
    LabelEncoderPair,
    build_classification_result,
    compute_classification_metrics,
    compute_inverse_class_weights,
)
from draft.config import AppConfig
from draft.schemas import ClassificationResult, IncomingEmail


class MultiTaskEmailClassifier(nn.Module):
    """Shared Transformer encoder with joint intent and urgency classification heads."""

    def __init__(
        self,
        backbone_name: str | None = None,
        num_intents: int = 10,
        num_urgencies: int = 3,
        dropout: float = 0.1,
        backbone_config: PretrainedConfig | None = None,
    ) -> None:
        super().__init__()
        if backbone_config is not None:
            self.backbone = AutoModel.from_config(backbone_config)
        elif backbone_name is not None:
            self.backbone = AutoModel.from_pretrained(backbone_name)
        else:
            raise ValueError("Provide either backbone_name or backbone_config.")

        hidden_size = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden_size, num_intents)
        self.urgency_head = nn.Linear(hidden_size, num_urgencies)

    @staticmethod
    def _masked_mean_pool(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool token representations while ignoring padding positions."""
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute joint (intent_logits, urgency_logits) for a tokenized batch."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._masked_mean_pool(outputs.last_hidden_state, attention_mask)
        dropped = self.dropout(pooled)
        return self.intent_head(dropped), self.urgency_head(dropped)


class EmailClassificationDataset(Dataset):
    """Torch dataset wrapping formatted email strings and integer task labels."""

    def __init__(
        self,
        texts: list[str],
        intent_ids: list[int] | None = None,
        urgency_ids: list[int] | None = None,
    ) -> None:
        self.texts = texts
        self.intent_ids = intent_ids
        self.urgency_ids = urgency_ids

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {"text": self.texts[idx]}
        if self.intent_ids is not None and self.urgency_ids is not None:
            item["intent_id"] = self.intent_ids[idx]
            item["urgency_id"] = self.urgency_ids[idx]
        return item


class _BatchCollator:
    """Tokenize a list of dataset examples into padded tensor batches."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        result: dict[str, torch.Tensor] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if "intent_id" in batch[0]:
            result["intent_ids"] = torch.tensor(
                [item["intent_id"] for item in batch], dtype=torch.long
            )
            result["urgency_ids"] = torch.tensor(
                [item["urgency_id"] for item in batch], dtype=torch.long
            )
        return result


class MultiTaskTrainer:
    """Trainer for joint intent and urgency fine-tuning with class-weighted loss."""

    def __init__(
        self,
        config: AppConfig,
        labels: LabelEncoderPair,
        model: MultiTaskEmailClassifier | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        device: str | None = None,
        use_amp: bool = True,
    ) -> None:
        self.config = config
        self.labels = labels
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_amp = use_amp
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(config.classifier.backbone_model)
        )
        self.model = (
            model
            if model is not None
            else MultiTaskEmailClassifier(
                backbone_name=config.classifier.backbone_model,
                num_intents=len(labels.intent_labels),
                num_urgencies=len(labels.urgency_labels),
            )
        )
        self.model.to(self.device)

    def _resolve_amp_dtype(self) -> torch.dtype | None:
        """Select safe mixed-precision dtype, guarding DeBERTa-v3 against FP16 overflow on T4."""
        if not self.use_amp or self.device.type != "cuda":
            return None
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        model_type = str(getattr(self.model.backbone.config, "model_type", "")).lower()
        backbone_name = self.config.classifier.backbone_model.lower()
        # DeBERTa-v3 disentangled attention overflows FP16 dynamic range on Turing (T4) GPUs.
        if "deberta" in model_type or "deberta" in backbone_name:
            return None
        return torch.float16

    def _make_dataloader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        """Create a padded DataLoader from a canonical split dataframe."""
        dataset = EmailClassificationDataset(
            texts=df["email"].astype(str).tolist(),
            intent_ids=[self.labels.encode_intent(val) for val in df["intent"]],
            urgency_ids=[self.labels.encode_urgency(val) for val in df["urgency"]],
        )
        collator = _BatchCollator(self.tokenizer, self.config.classifier.max_length)
        return DataLoader(
            dataset,
            batch_size=self.config.classifier.batch_size,
            shuffle=shuffle,
            collate_fn=collator,
        )

    def _build_loss_fns(
        self,
        train_df: pd.DataFrame,
    ) -> tuple[nn.CrossEntropyLoss, nn.CrossEntropyLoss]:
        """Build inverse-frequency weighted CrossEntropyLoss criteria for both heads."""
        intent_ids = [self.labels.encode_intent(val) for val in train_df["intent"]]
        urgency_ids = [self.labels.encode_urgency(val) for val in train_df["urgency"]]
        intent_weights = torch.tensor(
            compute_inverse_class_weights(intent_ids, len(self.labels.intent_labels)),
            dtype=torch.float32,
            device=self.device,
        )
        urgency_weights = torch.tensor(
            compute_inverse_class_weights(urgency_ids, len(self.labels.urgency_labels)),
            dtype=torch.float32,
            device=self.device,
        )
        return nn.CrossEntropyLoss(weight=intent_weights), nn.CrossEntropyLoss(
            weight=urgency_weights
        )

    def _train_epoch(
        self,
        loader: DataLoader,
        intent_loss_fn: nn.CrossEntropyLoss,
        urgency_loss_fn: nn.CrossEntropyLoss,
        optimizer: AdamW,
        scheduler: Any,
        amp_dtype: torch.dtype | None,
        scaler: torch.amp.GradScaler,
    ) -> float:
        """Run one training epoch and return mean joint loss."""
        self.model.train()
        total_loss = 0.0
        urgency_weight = self.config.classifier.urgency_loss_weight

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            intent_targets = batch["intent_ids"].to(self.device)
            urgency_targets = batch["urgency_ids"].to(self.device)

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                intent_logits, urgency_logits = self.model(input_ids, attention_mask)
                loss = intent_loss_fn(intent_logits, intent_targets) + (
                    urgency_weight * urgency_loss_fn(urgency_logits, urgency_targets)
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += float(loss.item())

        return total_loss / max(len(loader), 1)

    def _predict_split(self, loader: DataLoader) -> tuple[list[int], list[int]]:
        """Run evaluation inference over a DataLoader and return predicted IDs."""
        self.model.eval()
        pred_intents: list[int] = []
        pred_urgencies: list[int] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                intent_logits, urgency_logits = self.model(input_ids, attention_mask)
                pred_intents.extend(intent_logits.argmax(dim=-1).cpu().tolist())
                pred_urgencies.extend(urgency_logits.argmax(dim=-1).cpu().tolist())
        return pred_intents, pred_urgencies

    def _evaluate_split(self, df: pd.DataFrame, loader: DataLoader) -> dict[str, Any]:
        """Evaluate current model predictions against true labels of a split dataframe."""
        pred_intents, pred_urgencies = self._predict_split(loader)
        true_intents = [self.labels.encode_intent(val) for val in df["intent"]]
        true_urgencies = [self.labels.encode_urgency(val) for val in df["urgency"]]
        return compute_classification_metrics(
            true_intents,
            pred_intents,
            true_urgencies,
            pred_urgencies,
            self.labels,
        )

    def save_checkpoint(self, output_dir: Path) -> None:
        """Persist safetensors weights, backbone config, tokenizer, and metadata."""
        output_dir.mkdir(parents=True, exist_ok=True)
        save_model(self.model, str(output_dir / "model.safetensors"))
        self.model.backbone.config.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        meta = {
            "backbone_model": self.config.classifier.backbone_model,
            "max_length": self.config.classifier.max_length,
            "dropout": float(self.model.dropout.p),
            "labels": self.labels.to_dict(),
        }
        (output_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Train joint model, select best checkpoint by val combined macro-F1, and evaluate test."""
        torch.manual_seed(self.config.data.random_seed)
        train_loader = self._make_dataloader(train_df, shuffle=True)
        val_loader = self._make_dataloader(val_df, shuffle=False)
        test_loader = self._make_dataloader(test_df, shuffle=False)

        intent_loss_fn, urgency_loss_fn = self._build_loss_fns(train_df)
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.classifier.learning_rate,
            weight_decay=0.01,
        )
        epochs = self.config.classifier.epochs
        total_steps = max(len(train_loader) * epochs, 1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        amp_dtype = self._resolve_amp_dtype()
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

        best_score = -1.0
        best_epoch = 0
        best_val_metrics: dict[str, Any] = {}
        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(
                train_loader,
                intent_loss_fn,
                urgency_loss_fn,
                optimizer,
                scheduler,
                amp_dtype,
                scaler,
            )
            val_metrics = self._evaluate_split(val_df, val_loader)
            score = float(val_metrics["combined_macro_f1"])
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "val_intent_macro_f1": float(val_metrics["intent"]["macro_f1"]),
                    "val_urgency_macro_f1": float(val_metrics["urgency"]["macro_f1"]),
                    "val_combined_macro_f1": score,
                }
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_val_metrics = val_metrics
                self.save_checkpoint(output_dir)

        load_model(self.model, str(output_dir / "model.safetensors"), device=str(self.device))
        test_metrics = self._evaluate_split(test_df, test_loader)
        summary = {
            "backbone_model": self.config.classifier.backbone_model,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "history": history,
            "val": best_val_metrics,
            "test": test_metrics,
        }
        (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


class TransformerEmailClassifier:
    """Offline inference wrapper for a saved MultiTaskEmailClassifier checkpoint."""

    def __init__(
        self,
        model: MultiTaskEmailClassifier,
        tokenizer: PreTrainedTokenizerBase,
        labels: LabelEncoderPair,
        max_length: int = 256,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_length = max_length

    @classmethod
    def load(cls, model_dir: Path, device: str | None = None) -> "TransformerEmailClassifier":
        """Load saved checkpoint, config, and tokenizer locally without network calls."""
        meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
        labels = LabelEncoderPair.from_dict(meta["labels"])
        backbone_config = AutoConfig.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = MultiTaskEmailClassifier(
            num_intents=len(labels.intent_labels),
            num_urgencies=len(labels.urgency_labels),
            dropout=float(meta.get("dropout", 0.1)),
            backbone_config=backbone_config,
        )
        load_model(model, str(model_dir / "model.safetensors"), device="cpu")
        return cls(
            model=model,
            tokenizer=tokenizer,
            labels=labels,
            max_length=int(meta.get("max_length", 256)),
            device=device,
        )

    def predict_texts(self, texts: list[str], batch_size: int = 32) -> list[ClassificationResult]:
        """Classify raw formatted email strings into typed ClassificationResult objects."""
        if not texts:
            return []
        results: list[ClassificationResult] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                intent_logits, urgency_logits = self.model(input_ids, attention_mask)
                intent_probs = torch.softmax(intent_logits, dim=-1).cpu().tolist()
                urgency_probs = torch.softmax(urgency_logits, dim=-1).cpu().tolist()
                for i_probs, u_probs in zip(intent_probs, urgency_probs, strict=True):
                    results.append(build_classification_result(i_probs, u_probs, self.labels))
        return results

    def predict(self, email: IncomingEmail) -> ClassificationResult:
        """Classify a single incoming customer email."""
        return self.predict_texts([email.formatted_text])[0]

    def predict_batch(
        self,
        emails: list[IncomingEmail],
        batch_size: int = 32,
    ) -> list[ClassificationResult]:
        """Classify a batch of incoming customer emails."""
        return self.predict_texts([email.formatted_text for email in emails], batch_size=batch_size)

    def evaluate(self, df: pd.DataFrame, batch_size: int = 32) -> dict[str, Any]:
        """Evaluate checkpoint accuracy, F1 scores, and confusion matrices on a dataframe."""
        preds = self.predict_texts(df["email"].astype(str).tolist(), batch_size=batch_size)
        y_true_intent = [self.labels.encode_intent(val) for val in df["intent"]]
        y_pred_intent = [self.labels.encode_intent(p.intent) for p in preds]
        y_true_urgency = [self.labels.encode_urgency(val) for val in df["urgency"]]
        y_pred_urgency = [self.labels.encode_urgency(p.urgency) for p in preds]
        return compute_classification_metrics(
            y_true_intent,
            y_pred_intent,
            y_true_urgency,
            y_pred_urgency,
            self.labels,
        )
