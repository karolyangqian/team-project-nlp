"""Streamlit interactive demo UI for the DRAFT customer service email assistant."""

from pathlib import Path

import pandas as pd
import streamlit as st

from draft.config import AppConfig, load_config
from draft.generation.prompts import PromptBuilder
from draft.pipeline import DraftPipeline
from draft.retrieval.dual_retriever import DualSourceRetriever
from draft.schemas import IncomingEmail

PRESET_EMAILS: dict[str, tuple[str, str]] = {
    "Billing Dispute": (
        "Duplicate charge on monthly invoice",
        (
            "Hello Support Team, we noticed a duplicate line item on the monthly invoice "
            "issued on the first of this month. Could you review the overcharge and let us "
            "know when a refund can be issued?"
        ),
    ),
    "Service Outage": (
        "Account management portal offline for entire team",
        (
            "Our organization is locked out of the centralized account management portal "
            "across multiple browsers since 09:00 UTC. Please treat this as urgent and share "
            "the current outage status and update cadence."
        ),
    ),
    "MFA Reset": (
        "Admin lost authenticator phone",
        (
            "One of our workspace administrators lost their mobile device and used all backup "
            "codes. What verification is required for Support to reset MFA on their account?"
        ),
    ),
}


@st.cache_resource(show_spinner="Loading DRAFT models and retrieval index...")
def _load_shared_components(
    config_path: str,
    classifier_backend: str,
    enable_reranker: bool,
) -> tuple[AppConfig, object, DualSourceRetriever, PromptBuilder]:
    """Load and cache Stage 1 classifier, Stage 2 retriever, and Stage 3 prompt builder."""
    cfg = load_config(Path(config_path))
    classifier = DraftPipeline.load_classifier(cfg, backend=classifier_backend)
    retriever = DualSourceRetriever.load(cfg, enable_reranker=enable_reranker)
    prompt_builder = PromptBuilder(cfg.paths.prompts_dir)
    return cfg, classifier, retriever, prompt_builder


@st.cache_resource(show_spinner="Loading Hugging Face model and LoRA adapter...")
def _load_hf_generator(config_path: str, adapter_dir: str) -> object:
    """Load and cache the Hugging Face causal LLM generator across Streamlit reruns."""
    cfg = load_config(Path(config_path))
    resolved_adapter = Path(adapter_dir.strip()) if adapter_dir.strip() else None
    return DraftPipeline.load_generator(cfg, backend="hf", adapter_dir=resolved_adapter)


def _apply_preset(preset_name: str) -> None:
    """Populate subject and body session state from a preset email template."""
    subject, body = PRESET_EMAILS[preset_name]
    st.session_state["email_subject"] = subject
    st.session_state["email_body"] = body


def main() -> None:
    """Render Streamlit application for end-to-end DRAFT email reply generation."""
    st.set_page_config(page_title="DRAFT: Customer Service Email Assistant", layout="wide")
    st.title("DRAFT: Customer Service Email Assistant")
    st.caption(
        "Dual-source retrieval, intent and urgency classification, and grounded draft generation."
    )

    if "email_subject" not in st.session_state:
        _apply_preset("Billing Dispute")

    with st.sidebar:
        st.header("Configuration")
        config_path = st.text_input("Config Path", value="configs/default.yaml")
        classifier_backend = st.selectbox(
            "Stage 1 Classifier",
            options=["auto", "baseline", "transformer"],
            index=0,
        )
        generator_backend = st.selectbox(
            "Stage 3 Generator",
            options=["ollama", "hf", "mock", "openai"],
            index=0,
        )

        default_cfg = load_config(Path(config_path))
        api_base_url: str | None = None
        api_model: str | None = None
        api_key: str | None = None
        ollama_model = default_cfg.generation.ollama_model
        ollama_url = default_cfg.generation.ollama_base_url
        default_adapter = default_cfg.paths.artifacts_dir / "llm" / "lora_adapter"
        hf_adapter_dir = str(default_adapter) if default_adapter.exists() else ""

        if generator_backend == "ollama":
            ollama_options = list(dict.fromkeys([ollama_model, "qwen2.5:3b", "Custom..."]))
            selected_ollama = st.selectbox("Ollama Model", options=ollama_options, index=0)
            if selected_ollama == "Custom...":
                ollama_model = st.text_input("Custom Ollama Model", value=ollama_model)
            else:
                ollama_model = selected_ollama
            ollama_url = st.text_input("Ollama Base URL", value=ollama_url)
        elif generator_backend == "hf":
            st.caption(f"Base model: {default_cfg.generation.base_model}")
            hf_adapter_dir = st.text_input("LoRA Adapter Dir (optional)", value=hf_adapter_dir)
        elif generator_backend == "openai":
            api_base_url = st.text_input("API Base URL", value="http://localhost:8000/v1")
            api_model = st.text_input("API Model Name", value=default_cfg.generation.base_model)
            api_key = st.text_input("API Key (optional)", value="", type="password") or None

        use_rag = st.checkbox("Enable Dual-Source RAG", value=True)
        use_reranker = st.checkbox("Enable Cross-Encoder Reranker", value=True)
        use_few_shot = st.checkbox("Enable 2-Shot Policy Grounding", value=True)

        st.divider()
        st.subheader("Sample Emails")
        for preset_name in PRESET_EMAILS:
            if st.button(preset_name, use_container_width=True):
                _apply_preset(preset_name)

    subject_input = st.text_input("Email Subject", key="email_subject")
    body_input = st.text_area("Customer Email Body", key="email_body", height=160)

    if not st.button("Generate Draft", type="primary"):
        return

    if not body_input.strip():
        st.warning("Please enter a customer email body before generating a draft.")
        return

    try:
        cfg, classifier, retriever, prompt_builder = _load_shared_components(
            config_path=config_path,
            classifier_backend=classifier_backend,
            enable_reranker=use_reranker,
        )
        cfg.generation.ollama_model = ollama_model
        cfg.generation.ollama_base_url = ollama_url
        if generator_backend == "hf":
            generator = _load_hf_generator(config_path=config_path, adapter_dir=hf_adapter_dir)
        else:
            generator = DraftPipeline.load_generator(
                cfg,
                backend=generator_backend,
                api_base_url=api_base_url,
                api_model=api_model,
                api_key=api_key,
            )
        pipeline = DraftPipeline(cfg, classifier, retriever, prompt_builder, generator)

        with st.spinner("Running classification, dual-source retrieval, and draft generation..."):
            draft, timings = pipeline.run_with_timing(
                email=IncomingEmail(subject=subject_input, body=body_input),
                use_rag=use_rag,
                use_reranker=use_reranker,
                use_few_shot=use_few_shot,
            )
    except Exception as exc:
        st.error(f"Pipeline execution failed: {exc}")
        return

    st.subheader("Stage 1: Intent and Urgency Classification")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Intent", str(draft.classification.intent))
    c2.metric("Intent Confidence", f"{draft.classification.intent_confidence:.2%}")
    c3.metric("Predicted Urgency", str(draft.classification.urgency).upper())
    c4.metric("Urgency Confidence", f"{draft.classification.urgency_confidence:.2%}")

    with st.expander("Intent and Urgency Probability Breakdown"):
        intent_rows = [
            {"Intent": k, "Probability": round(v, 4)}
            for k, v in sorted(
                draft.classification.intent_scores.items(), key=lambda kv: kv[1], reverse=True
            )
        ]
        urgency_rows = [
            {"Urgency": k, "Probability": round(v, 4)}
            for k, v in sorted(
                draft.classification.urgency_scores.items(), key=lambda kv: kv[1], reverse=True
            )
        ]
        col_i, col_u = st.columns(2)
        col_i.dataframe(pd.DataFrame(intent_rows), hide_index=True, use_container_width=True)
        col_u.dataframe(pd.DataFrame(urgency_rows), hide_index=True, use_container_width=True)

    st.subheader("Stage 3: Generated Email Draft")
    st.text_input("Reply Subject", value=draft.subject, disabled=True)
    cited_text = ", ".join(draft.cited_doc_ids) if draft.cited_doc_ids else "None"
    st.caption(f"Model: {draft.model_name} | Cited Policy Docs: {cited_text}")
    st.text_area("Drafted Reply Body", value=draft.body, height=220)

    latency_rows = [
        {"Stage": "Stage 1 (Classifier)", "Latency (ms)": round(timings["stage1_ms"], 1)},
        {"Stage": "Stage 2 (Retrieval)", "Latency (ms)": round(timings["stage2_ms"], 1)},
        {"Stage": "Stage 3 (Generation)", "Latency (ms)": round(timings["stage3_ms"], 1)},
        {"Stage": "Total", "Latency (ms)": round(timings["total_ms"], 1)},
    ]
    st.dataframe(pd.DataFrame(latency_rows), hide_index=True, use_container_width=True)

    st.subheader("Stage 2: Retrieved Context")
    tab_docs, tab_refs = st.tabs(["Organization Policy Docs", "Reference Replies"])
    with tab_docs:
        if not draft.retrieved_context.top_k_docs:
            st.info("No policy documents retrieved (RAG disabled or empty index).")
        for doc in draft.retrieved_context.top_k_docs:
            st.markdown(
                f"**[{doc.doc_id}] {doc.title}** (`{doc.category}`, score=`{doc.score:.4f}`)"
            )
            st.write(doc.content)
            st.divider()

    with tab_refs:
        if not draft.retrieved_context.top_k_responses:
            st.info("No reference replies retrieved (RAG disabled or empty index).")
        for idx, resp in enumerate(draft.retrieved_context.top_k_responses, start=1):
            st.markdown(
                f"**Example {idx}: {resp.email_id}** "
                f"(`{resp.intent}` | `{resp.urgency}`, score=`{resp.score:.4f}`)"
            )
            st.caption(f"Customer Email: {resp.email[:220]}...")
            st.write(resp.answer)
            st.divider()


if __name__ == "__main__":
    main()
