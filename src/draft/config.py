"""Typed YAML configuration schema and loader."""

import math
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from draft.schemas import DocCategory, IntentLabel, UrgencyLevel


class PathsConfig(BaseModel):
    """Workspace paths for data, models, prompts, and reports."""

    raw_dir: Path
    processed_dir: Path
    org_docs_file: Path
    artifacts_dir: Path
    prompts_dir: Path
    reports_dir: Path


class DataConfig(BaseModel):
    """Dataset source, filtering thresholds, and canonical split settings."""

    hf_dataset_id: str
    language: str = "en"
    train_ratio: float = Field(gt=0.0, lt=1.0)
    val_ratio: float = Field(gt=0.0, lt=1.0)
    test_ratio: float = Field(gt=0.0, lt=1.0)
    random_seed: int = 42
    min_email_chars: int = Field(default=20, ge=1)
    min_answer_chars: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def validate_split_ratios(self) -> "DataConfig":
        """Ensure train, val, and test ratios sum to 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")
        return self


class TaxonomyConfig(BaseModel):
    """Label taxonomy and intent-to-document-category mapping."""

    intents: list[IntentLabel]
    urgencies: list[UrgencyLevel]
    doc_categories: list[DocCategory]
    intent_to_doc_categories: dict[IntentLabel, list[DocCategory]]

    @model_validator(mode="after")
    def validate_mappings(self) -> "TaxonomyConfig":
        """Ensure every configured intent maps to at least one configured doc category."""
        configured_intents = set(self.intents)
        configured_docs = set(self.doc_categories)
        if set(IntentLabel) != configured_intents:
            raise ValueError("Configured intents must match IntentLabel enum members.")
        if set(DocCategory) != configured_docs:
            raise ValueError("Configured doc_categories must match DocCategory enum members.")

        missing_intents = configured_intents - set(self.intent_to_doc_categories.keys())
        if missing_intents:
            raise ValueError(f"Missing intent_to_doc_categories entries for: {missing_intents}")

        for intent, categories in self.intent_to_doc_categories.items():
            if not categories:
                raise ValueError(f"Intent '{intent}' must map to at least one doc category.")
            invalid = set(categories) - configured_docs
            if invalid:
                raise ValueError(f"Intent '{intent}' maps to invalid doc categories: {invalid}")
        return self


class ClassifierConfig(BaseModel):
    """Stage 1 multi-task classifier hyperparameters."""

    backbone_model: str
    max_length: int = Field(ge=32)
    batch_size: int = Field(ge=1)
    epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    urgency_loss_weight: float = Field(gt=0.0)


class RetrievalConfig(BaseModel):
    """Stage 2 dual-source retrieval and reranking configuration."""

    embedding_model: str
    reranker_model: str
    top_k_docs: int = Field(ge=1)
    top_k_responses: int = Field(ge=1)
    dense_weight: float = Field(ge=0.0, le=1.0)
    bm25_weight: float = Field(ge=0.0, le=1.0)
    intent_boost: float = Field(ge=0.0)
    rrf_k: int = Field(ge=1)


class GenerationConfig(BaseModel):
    """Stage 3 LLM generation and local serving settings."""

    base_model: str
    ollama_model: str
    ollama_base_url: str
    max_new_tokens: int = Field(ge=16)
    temperature: float = Field(ge=0.0)


class AppConfig(BaseModel):
    """Root application configuration."""

    paths: PathsConfig
    data: DataConfig
    taxonomy: TaxonomyConfig
    classifier: ClassifierConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig


def load_config(config_path: Path | str = "configs/default.yaml") -> AppConfig:
    """Load and validate application configuration from a YAML file."""
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(raw)
