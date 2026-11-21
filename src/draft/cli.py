"""Command-line interface for DRAFT pipeline stages."""

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from draft.classifier.base import LabelEncoderPair
from draft.classifier.baseline import TfidfBaselineClassifier
from draft.config import AppConfig, load_config
from draft.data.acquisition import DataAcquisitionService
from draft.data.org_docs import load_org_docs
from draft.generation.backends import (
    MockGenerator,
    OllamaGenerator,
    OpenAICompatibleGenerator,
)
from draft.generation.prompts import PromptBuilder, build_reply_subject, extract_cited_doc_ids
from draft.generation.sft_dataset import SftDatasetBuilder
from draft.retrieval.dual_retriever import DualSourceRetriever
from draft.retrieval.encoders import SentenceTransformerEncoder
from draft.retrieval.metrics import run_retrieval_ablations
from draft.schemas import DraftResponse, IncomingEmail, IntentLabel

app = typer.Typer(help="DRAFT: Dual-source RAG and classification CLI.")


class ClassifierBackend(StrEnum):
    """Available Stage 1 classifier backends."""

    BASELINE = "baseline"
    TRANSFORMER = "transformer"


class GeneratorBackend(StrEnum):
    """Available Stage 3 draft generator backends."""

    MOCK = "mock"
    OLLAMA = "ollama"
    HF = "hf"
    OPENAI = "openai"


class EvalSplit(StrEnum):
    """Evaluation splits supported by CLI evaluation commands."""

    VAL = "val"
    TEST = "test"


@app.callback()
def main() -> None:
    """CLI entrypoint for DRAFT."""


def _echo_split_metrics(label: str, metrics: dict) -> None:
    """Print concise Stage 1 evaluation summary for one split."""
    intent = metrics["intent"]
    urgency = metrics["urgency"]
    typer.echo(
        f"  [{label}] "
        f"intent(acc={intent['accuracy']:.4f}, macro_f1={intent['macro_f1']:.4f}, "
        f"weighted_f1={intent['weighted_f1']:.4f}) | "
        f"urgency(acc={urgency['accuracy']:.4f}, macro_f1={urgency['macro_f1']:.4f}, "
        f"weighted_f1={urgency['weighted_f1']:.4f})"
    )


def _load_stage1_classifier(cfg: AppConfig, backend: ClassifierBackend):
    """Load the configured Stage 1 classifier backend."""
    if backend == ClassifierBackend.BASELINE:
        return TfidfBaselineClassifier.load(cfg.paths.artifacts_dir / "classifier_baseline")
    from draft.classifier.multitask import TransformerEmailClassifier

    return TransformerEmailClassifier.load(cfg.paths.artifacts_dir / "classifier")


def _load_stage3_generator(
    cfg: AppConfig,
    backend: GeneratorBackend,
    adapter_dir: Path | None,
    api_base_url: str | None = None,
    api_model: str | None = None,
    api_key: str | None = None,
):
    """Instantiate the configured Stage 3 generator backend."""
    if backend == GeneratorBackend.MOCK:
        return MockGenerator()
    if backend == GeneratorBackend.OLLAMA:
        return OllamaGenerator(
            model_name=cfg.generation.ollama_model,
            base_url=cfg.generation.ollama_base_url,
            default_max_new_tokens=cfg.generation.max_new_tokens,
            default_temperature=cfg.generation.temperature,
        )
    if backend == GeneratorBackend.OPENAI:
        return OpenAICompatibleGenerator(
            model_name=api_model or cfg.generation.base_model,
            base_url=api_base_url or "http://localhost:8000/v1",
            api_key=api_key,
            default_max_new_tokens=cfg.generation.max_new_tokens,
            default_temperature=cfg.generation.temperature,
        )
    from draft.generation.hf_backend import HFPeftGenerator

    default_adapter = cfg.paths.artifacts_dir / "llm" / "lora_adapter"
    resolved_adapter = (
        adapter_dir
        if adapter_dir is not None
        else (default_adapter if default_adapter.exists() else None)
    )
    return HFPeftGenerator(
        base_model_name=cfg.generation.base_model,
        adapter_dir=resolved_adapter,
        default_max_new_tokens=cfg.generation.max_new_tokens,
        default_temperature=cfg.generation.temperature,
    )


@app.command("prepare-data")
def prepare_data(
    config: Annotated[
        Path, typer.Option(help="Path to YAML config file.")
    ] = Path("configs/default.yaml"),
    force_download: Annotated[
        bool, typer.Option(help="Re-download raw Hugging Face dataset.")
    ] = False,
    overwrite_org_docs: Annotated[
        bool, typer.Option(help="Regenerate default organization docs.")
    ] = False,
) -> None:
    """Download, clean, split datasets and write the EDA summary report."""
    app_config = load_config(config)
    summary = DataAcquisitionService(app_config).run(
        force_download=force_download,
        overwrite_org_docs=overwrite_org_docs,
    )

    stats = summary["cleaning_stats"]
    splits = summary["split_sizes"]
    typer.echo("Data preparation completed:")
    typer.echo(
        f"  Raw rows: {stats['raw_rows']} -> EN: {stats['after_language_filter']} "
        f"-> Valid labels: {stats['after_label_filter']} "
        f"-> Deduplicated: {stats['after_deduplication']}"
    )
    typer.echo(
        f"  Splits: train={splits['train']}, val={splits['val']}, test={splits['test']}"
    )
    typer.echo(f"  Org docs: {summary['org_docs_total']}")
    for label, path in summary["output_paths"].items():
        typer.echo(f"  [{label}] {path}")


@app.command("train-baseline")
def train_baseline(
    config: Annotated[
        Path, typer.Option(help="Path to YAML config file.")
    ] = Path("configs/default.yaml"),
) -> None:
    """Train and evaluate the local TF-IDF + LogisticRegression baseline."""
    app_config = load_config(config)
    processed = app_config.paths.processed_dir
    train_df = pd.read_parquet(processed / "train.parquet")
    val_df = pd.read_parquet(processed / "val.parquet")
    test_df = pd.read_parquet(processed / "test.parquet")

    labels = LabelEncoderPair.from_taxonomy(app_config.taxonomy)
    clf = TfidfBaselineClassifier(labels=labels, random_state=app_config.data.random_seed)
    clf.fit(train_df)

    model_dir = app_config.paths.artifacts_dir / "classifier_baseline"
    clf.save(model_dir)

    summary = {"val": clf.evaluate(val_df), "test": clf.evaluate(test_df)}
    app_config.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = app_config.paths.reports_dir / "classifier_baseline_metrics.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    typer.echo("TF-IDF baseline training completed:")
    _echo_split_metrics("val", summary["val"])
    _echo_split_metrics("test", summary["test"])
    typer.echo(f"  [model] {model_dir}")
    typer.echo(f"  [report] {report_path}")


@app.command("classify")
def classify(
    body: Annotated[str, typer.Option(help="Customer email body text.")],
    subject: Annotated[str, typer.Option(help="Customer email subject line.")] = "",
    backend: Annotated[
        ClassifierBackend, typer.Option(help="Stage 1 classifier backend.")
    ] = ClassifierBackend.BASELINE,
    config: Annotated[
        Path, typer.Option(help="Path to YAML config file.")
    ] = Path("configs/default.yaml"),
) -> None:
    """Predict intent and urgency for a single customer email."""
    app_config = load_config(config)
    classifier = _load_stage1_classifier(app_config, backend)
    result = classifier.predict(IncomingEmail(subject=subject, body=body))
    top_intents = sorted(result.intent_scores.items(), key=lambda kv: kv[1], reverse=True)[:3]

    typer.echo(f"Intent:  {result.intent} (confidence={result.intent_confidence:.4f})")
    typer.echo(f"Urgency: {result.urgency} (confidence={result.urgency_confidence:.4f})")
    typer.echo("Top-3 intent probabilities:")
    for label, prob in top_intents:
        typer.echo(f"  - {label}: {prob:.4f}")


@app.command("build-index")
def build_index(
    config: Annotated[
        Path, typer.Option(help="Path to YAML config file.")
    ] = Path("configs/default.yaml"),
    encoder_path: Annotated[
        Path | None, typer.Option(help="Optional fine-tuned bi-encoder directory.")
    ] = None,
    output_dir: Annotated[
        Path | None, typer.Option(help="Target directory for dual retrieval indices.")
    ] = None,
    max_ref_emails: Annotated[
        int | None, typer.Option(help="Optional cap on indexed training emails.")
    ] = None,
) -> None:
    """Encode organization docs and train emails into dual FAISS + BM25 indices."""
    cfg = load_config(config)
    train_df = pd.read_parquet(cfg.paths.processed_dir / "train.parquet")
    if max_ref_emails is not None and max_ref_emails < len(train_df):
        train_df = train_df.head(max_ref_emails).reset_index(drop=True)
    org_docs = load_org_docs(cfg.paths.org_docs_file)

    default_ft_dir = cfg.paths.artifacts_dir / "retriever" / "encoder"
    model_ref: str | Path = (
        encoder_path
        if encoder_path is not None
        else (default_ft_dir if default_ft_dir.exists() else cfg.retrieval.embedding_model)
    )
    encoder = SentenceTransformerEncoder(model_ref, show_progress_bar=True)
    saved_dir = DualSourceRetriever.build_and_save_indices(
        config=cfg,
        encoder=encoder,
        train_df=train_df,
        org_docs=org_docs,
        output_dir=output_dir,
    )
    meta = json.loads((saved_dir / "index_meta.json").read_text(encoding="utf-8"))
    typer.echo("Dual retrieval index built:")
    typer.echo(f"  Encoder: {meta['embedding_model']} (dim={meta['embedding_dim']})")
    typer.echo(f"  Org docs indexed: {meta['org_docs_count']}")
    typer.echo(f"  Ref emails indexed: {meta['ref_emails_count']}")
    typer.echo(f"  [index] {saved_dir}")


@app.command("retrieve")
def retrieve(
    body: Annotated[str, typer.Option(help="Customer email body text.")],
    subject: Annotated[str, typer.Option(help="Customer email subject line.")] = "",
    intent: Annotated[
        IntentLabel | None, typer.Option(help="Optional intent label (auto-classified if omitted).")
    ] = None,
    classifier_backend: Annotated[
        ClassifierBackend, typer.Option(help="Classifier backend when intent is omitted.")
    ] = ClassifierBackend.BASELINE,
    rerank: Annotated[
        bool, typer.Option(help="Apply Cross-Encoder reranking on retrieved org docs.")
    ] = True,
    top_k_docs: Annotated[int | None, typer.Option(help="Number of org docs to return.")] = None,
    top_k_responses: Annotated[
        int | None, typer.Option(help="Number of reference replies to return.")
    ] = None,
    config: Annotated[
        Path, typer.Option(help="Path to YAML config file.")
    ] = Path("configs/default.yaml"),
) -> None:
    """Retrieve Top-K organization docs and Top-K reference responses for an email."""
    cfg = load_config(config)
    email = IncomingEmail(subject=subject, body=body)
    active_intent = intent
    if active_intent is None:
        clf_result = _load_stage1_classifier(cfg, classifier_backend).predict(email)
        active_intent = clf_result.intent
        typer.echo(
            f"Stage 1 Prediction: intent={clf_result.intent}, urgency={clf_result.urgency}"
        )

    retriever = DualSourceRetriever.load(cfg, enable_reranker=rerank)
    context = retriever.retrieve(
        email=email,
        intent=active_intent,
        top_k_docs=top_k_docs,
        top_k_responses=top_k_responses,
        use_reranker=rerank,
    )

    typer.echo("\nTop-K Organization Policy Docs:")
    for doc in context.top_k_docs:
        snippet = doc.content[:140].replace("\n", " ")
        typer.echo(f"  [{doc.doc_id}] ({doc.category}, score={doc.score:.4f}) {doc.title}")
        typer.echo(f"    {snippet}...")

    typer.echo("\nTop-K Reference Responses:")
    for resp in context.top_k_responses:
        snippet = resp.answer[:140].replace("\n", " ")
        typer.echo(
            f"  [{resp.email_id}] ({resp.intent} / {resp.urgency}, score={resp.score:.4f})"
        )
        typer.echo(f"    {snippet}...")


@app.command("eval-retrieval")
def eval_retrieval(
    split: Annotated[EvalSplit, typer.Option(help="Split to evaluate.")] = EvalSplit.VAL,
    max_queries: Annotated[int, typer.Option(help="Maximum queries to evaluate.")] = 300,
    rerank: Annotated[
        bool, typer.Option(help="Include Cross-Encoder reranking ablation on org docs.")
    ] = True,
    config: Annotated[
        Path, typer.Option(help="Path to YAML config file.")
    ] = Path("configs/default.yaml"),
) -> None:
    """Run retrieval ablation benchmarks and write reports/retrieval_metrics.json."""
    cfg = load_config(config)
    eval_df = pd.read_parquet(cfg.paths.processed_dir / f"{split.value}.parquet")
    retriever = DualSourceRetriever.load(cfg, enable_reranker=rerank)
    results = run_retrieval_ablations(
        retriever=retriever,
        eval_df=eval_df,
        max_queries=max_queries,
        include_reranker=rerank,
        seed=cfg.data.random_seed,
    )

    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.paths.reports_dir / "retrieval_metrics.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    typer.echo(f"Retrieval ablations on {split.value} ({results['num_queries']} queries):")
    typer.echo("  Organization Docs:")
    for mode, m in results["org_docs"].items():
        typer.echo(
            f"    {mode:25s} hit@1={m['category_hit@1']:.4f} "
            f"hit@3={m['category_hit@3']:.4f} mrr@3={m['category_mrr@3']:.4f} "
            f"unique_docs={m['unique_docs_retrieved']}"
        )
    typer.echo("  Reference Emails:")
    for mode, m in results["ref_emails"].items():
        tag_h3 = m.get("tag_hit@3", 0.0)
        typer.echo(
            f"    {mode:25s} intent_hit@1={m['intent_hit@1']:.4f} "
            f"intent_hit@3={m['intent_hit@3']:.4f} intent_mrr@3={m['intent_mrr@3']:.4f} "
            f"tag_hit@3={tag_h3:.4f}"
        )
    typer.echo(f"  [report] {report_path}")


@app.command("build-sft-data")
def build_sft_data(
    config: Annotated[
        Path, typer.Option(help="Path to YAML config file.")
    ] = Path("configs/default.yaml"),
    num_ticket_samples: Annotated[
        int, typer.Option(help="Number of stratified training tickets to include.")
    ] = 3000,
    policy_upsample: Annotated[
        int, typer.Option(help="Replication factor for policy-grounded examples.")
    ] = 3,
    output_dir: Annotated[
        Path | None, typer.Option(help="Output directory for SFT JSONL files.")
    ] = None,
    rerank: Annotated[
        bool, typer.Option(help="Apply Cross-Encoder reranking while building SFT context.")
    ] = False,
) -> None:
    """Build leakage-free RAG SFT datasets from train tickets and organization docs."""
    cfg = load_config(config)
    train_df = pd.read_parquet(cfg.paths.processed_dir / "train.parquet")
    org_docs = load_org_docs(cfg.paths.org_docs_file)
    retriever = DualSourceRetriever.load(cfg, enable_reranker=rerank)
    prompt_builder = PromptBuilder(cfg.paths.prompts_dir)

    summary = SftDatasetBuilder(cfg, retriever, prompt_builder).build(
        train_df=train_df,
        org_docs=org_docs,
        output_dir=output_dir,
        num_ticket_samples=num_ticket_samples,
        policy_upsample=policy_upsample,
        use_reranker=rerank,
    )
    typer.echo("RAG SFT dataset built:")
    typer.echo(
        f"  Total={summary['total_examples']} (train={summary['train_examples']}, "
        f"eval={summary['eval_examples']})"
    )
    typer.echo(
        f"  Sources: tickets={summary['ticket_examples']}, "
        f"policy_grounded={summary['policy_grounded_examples']} "
        f"(upsample={summary['policy_upsample']})"
    )
    typer.echo(f"  Prompt chars p50/p90/max: {summary['prompt_chars']}")
    for label, path in summary["output_paths"].items():
        typer.echo(f"  [{label}] {path}")


@app.command("generate-draft")
def generate_draft(
    body: Annotated[str, typer.Option(help="Customer email body text.")],
    subject: Annotated[str, typer.Option(help="Customer email subject line.")] = "",
    classifier_backend: Annotated[
        ClassifierBackend, typer.Option(help="Stage 1 classifier backend.")
    ] = ClassifierBackend.BASELINE,
    generator_backend: Annotated[
        GeneratorBackend, typer.Option(help="Stage 3 generator backend.")
    ] = GeneratorBackend.MOCK,
    adapter_dir: Annotated[
        Path | None, typer.Option(help="Optional LoRA adapter directory for HF backend.")
    ] = None,
    rerank: Annotated[
        bool, typer.Option(help="Apply Cross-Encoder reranking on retrieved org docs.")
    ] = True,
    config: Annotated[
        Path, typer.Option(help="Path to YAML config file.")
    ] = Path("configs/default.yaml"),
) -> None:
    """Run all three DRAFT stages to classify, retrieve context, and draft a reply."""
    cfg = load_config(config)
    email = IncomingEmail(subject=subject, body=body)
    clf_result = _load_stage1_classifier(cfg, classifier_backend).predict(email)
    retriever = DualSourceRetriever.load(cfg, enable_reranker=rerank)
    context = retriever.retrieve(email=email, intent=clf_result.intent, use_reranker=rerank)

    prompt_builder = PromptBuilder(cfg.paths.prompts_dir)
    messages = prompt_builder.render(email, clf_result, context)
    generator = _load_stage3_generator(cfg, generator_backend, adapter_dir)
    draft_body = generator.generate(messages)

    valid_ids = {doc.doc_id for doc in context.top_k_docs}
    draft = DraftResponse(
        subject=build_reply_subject(email.subject),
        body=draft_body,
        classification=clf_result,
        retrieved_context=context,
        cited_doc_ids=extract_cited_doc_ids(draft_body, valid_ids),
        model_name=generator.model_name,
    )
    typer.echo(
        f"Stage 1: intent={draft.classification.intent} "
        f"({draft.classification.intent_confidence:.4f}), "
        f"urgency={draft.classification.urgency} "
        f"({draft.classification.urgency_confidence:.4f})"
    )
    typer.echo(
        f"Stage 2 Docs: {[doc.doc_id for doc in draft.retrieved_context.top_k_docs]}"
    )
    typer.echo(
        f"Stage 2 Refs: {[resp.email_id for resp in draft.retrieved_context.top_k_responses]}"
    )
    typer.echo(f"Cited Docs:   {draft.cited_doc_ids}")
    typer.echo(f"Model:        {draft.model_name}")
    typer.echo(f"\nSubject: {draft.subject}\n\n{draft.body}")
