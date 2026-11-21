"""End-to-end DraftPipeline orchestrator unifying Stages 1, 2, and 3."""

import time
from pathlib import Path

from draft.classifier.base import EmailClassifier
from draft.classifier.baseline import TfidfBaselineClassifier
from draft.config import AppConfig
from draft.generation.backends import (
    BaseGenerator,
    MockGenerator,
    OllamaGenerator,
    OpenAICompatibleGenerator,
)
from draft.generation.prompts import PromptBuilder, build_reply_subject, extract_cited_doc_ids
from draft.retrieval.dual_retriever import DualSourceRetriever
from draft.schemas import DraftResponse, IncomingEmail, RetrievedContext


class DraftPipeline:
    """Orchestrates email classification, dual-source retrieval, and draft generation."""

    def __init__(
        self,
        config: AppConfig,
        classifier: EmailClassifier,
        retriever: DualSourceRetriever,
        prompt_builder: PromptBuilder,
        generator: BaseGenerator,
    ) -> None:
        self.config = config
        self.classifier = classifier
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.generator = generator

    def run_with_timing(
        self,
        email: IncomingEmail,
        use_rag: bool = True,
        use_reranker: bool = True,
        use_few_shot: bool = True,
        exclude_email_ids: set[str] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[DraftResponse, dict[str, float]]:
        """Execute all three pipeline stages and return the draft plus per-stage timings in ms."""
        t0 = time.perf_counter()
        classification = self.classifier.predict(email)
        t1 = time.perf_counter()

        if use_rag:
            context = self.retriever.retrieve(
                email=email,
                intent=classification.intent,
                exclude_email_ids=exclude_email_ids,
                use_reranker=use_reranker,
            )
        else:
            context = RetrievedContext()
        t2 = time.perf_counter()

        messages = self.prompt_builder.render(
            email,
            classification,
            context,
            use_few_shot=use_few_shot,
        )
        draft_body = self.generator.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        t3 = time.perf_counter()

        valid_doc_ids = {doc.doc_id for doc in context.top_k_docs}
        draft = DraftResponse(
            subject=build_reply_subject(email.subject),
            body=draft_body,
            classification=classification,
            retrieved_context=context,
            cited_doc_ids=extract_cited_doc_ids(draft_body, valid_doc_ids),
            model_name=self.generator.model_name,
        )
        timings = {
            "stage1_ms": (t1 - t0) * 1000.0,
            "stage2_ms": (t2 - t1) * 1000.0,
            "stage3_ms": (t3 - t2) * 1000.0,
            "total_ms": (t3 - t0) * 1000.0,
        }
        return draft, timings

    def run(
        self,
        email: IncomingEmail,
        use_rag: bool = True,
        use_reranker: bool = True,
        use_few_shot: bool = True,
        exclude_email_ids: set[str] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> DraftResponse:
        """Execute the full DRAFT pipeline and return a DraftResponse."""
        draft, _ = self.run_with_timing(
            email=email,
            use_rag=use_rag,
            use_reranker=use_reranker,
            use_few_shot=use_few_shot,
            exclude_email_ids=exclude_email_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return draft

    @classmethod
    def load_classifier(cls, config: AppConfig, backend: str = "auto") -> EmailClassifier:
        """Load the configured Stage 1 classifier backend."""
        mode = backend.lower()
        transformer_dir = config.paths.artifacts_dir / "classifier"
        baseline_dir = config.paths.artifacts_dir / "classifier_baseline"

        if mode == "auto":
            mode = (
                "transformer"
                if (transformer_dir / "model.safetensors").exists()
                else "baseline"
            )
        if mode == "baseline":
            return TfidfBaselineClassifier.load(baseline_dir)
        if mode == "transformer":
            from draft.classifier.multitask import TransformerEmailClassifier

            return TransformerEmailClassifier.load(transformer_dir)
        raise ValueError(f"Unsupported classifier backend: {backend}")

    @classmethod
    def load_generator(
        cls,
        config: AppConfig,
        backend: str = "ollama",
        adapter_dir: Path | None = None,
        ollama_model: str | None = None,
        api_base_url: str | None = None,
        api_model: str | None = None,
        api_key: str | None = None,
    ) -> BaseGenerator:
        """Instantiate the requested Stage 3 generator backend."""
        mode = backend.lower()
        if mode == "mock":
            return MockGenerator()
        if mode == "ollama":
            return OllamaGenerator(
                model_name=ollama_model or config.generation.ollama_model,
                base_url=config.generation.ollama_base_url,
                default_max_new_tokens=config.generation.max_new_tokens,
                default_temperature=config.generation.temperature,
            )
        if mode == "openai":
            return OpenAICompatibleGenerator(
                model_name=api_model or config.generation.base_model,
                base_url=api_base_url or "http://localhost:8000/v1",
                api_key=api_key,
                default_max_new_tokens=config.generation.max_new_tokens,
                default_temperature=config.generation.temperature,
            )
        if mode == "hf":
            from draft.generation.hf_backend import HFPeftGenerator

            default_adapter = config.paths.artifacts_dir / "llm" / "lora_adapter"
            resolved_adapter = (
                adapter_dir
                if adapter_dir is not None
                else (default_adapter if default_adapter.exists() else None)
            )
            return HFPeftGenerator(
                base_model_name=config.generation.base_model,
                adapter_dir=resolved_adapter,
                default_max_new_tokens=config.generation.max_new_tokens,
                default_temperature=config.generation.temperature,
            )
        raise ValueError(f"Unsupported generator backend: {backend}")

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        classifier_backend: str = "auto",
        generator_backend: str = "ollama",
        adapter_dir: Path | None = None,
        enable_reranker: bool = True,
        ollama_model: str | None = None,
        api_base_url: str | None = None,
        api_model: str | None = None,
        api_key: str | None = None,
    ) -> "DraftPipeline":
        """Build a full DraftPipeline from YAML configuration and saved artifacts."""
        classifier = cls.load_classifier(config, backend=classifier_backend)
        retriever = DualSourceRetriever.load(config, enable_reranker=enable_reranker)
        prompt_builder = PromptBuilder(config.paths.prompts_dir)
        generator = cls.load_generator(
            config,
            backend=generator_backend,
            adapter_dir=adapter_dir,
            ollama_model=ollama_model,
            api_base_url=api_base_url,
            api_model=api_model,
            api_key=api_key,
        )
        return cls(
            config=config,
            classifier=classifier,
            retriever=retriever,
            prompt_builder=prompt_builder,
            generator=generator,
        )
