# DRAFT: Dual-Source Retrieval, Intent/Urgency Analysis, Fine-Tuned LLM, Tailored Style

DRAFT is an end-to-end NLP and dual-source RAG system for automated customer service email draft generation. Given an incoming email, the pipeline classifies intent and urgency, retrieves both factual organizational policy documents and historical reference replies, and drafts a grounded, tone-matched response.

## Architecture Overview (3-Stage BEAM Workflow)

| Stage | Module | Role | Models and Methods |
| :--- | :--- | :--- | :--- |
| Stage 1: Text Classification | `src/draft/classifier/` | Predict `intent` (10 classes) and `urgency` (`low`, `medium`, `high`) | TF-IDF + class-weighted `LogisticRegression` baseline; Multi-Task `microsoft/deberta-v3-small` with inverse-frequency loss weighting |
| Stage 2: Dual-Source Retrieval | `src/draft/retrieval/` | Retrieve Top-K Organization Policy Docs (42 rules/facts) and Top-K Historical Reference Replies (18,959 train tickets) | `BAAI/bge-small-en-v1.5` (contrastive fine-tuning) + `faiss-cpu` dense + `rank-bm25` lexical via Reciprocal Rank Fusion + multiplicative soft intent boost + optional `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker |
| Stage 3: Draft Generation | `src/draft/generation/` | Combine incoming email, predicted intent/urgency, policy docs, and reference replies into a grounded draft reply | Jinja2 prompt templates + `Qwen/Qwen2.5-3B-Instruct` fine-tuned on RAG-formatted SFT examples via LoRA/QLoRA; served locally via Ollama GGUF or Hugging Face |
| Orchestration & Evaluation | `src/draft/pipeline.py`, `src/draft/evaluation/` | Unified timed execution and quantitative ablation reporting | ROUGE-1/2/L, BERTScore, citation precision, policy fact recall, and per-stage latency |

## Deployment Split (Local Laptop vs. Colab GPU)

| Component | Recommended Runtime | Notes |
| :--- | :--- | :--- |
| Data acquisition, cleaning, canonical splitting | Local Laptop (CPU) | Processes `61,765` HF tickets into `23,699` deduplicated English rows (`18,959` train / `2,370` val / `2,370` test) |
| Stage 1 TF-IDF baseline | Local Laptop (CPU) | Trains and evaluates in under 30 seconds |
| Stage 1 Multi-Task DeBERTa-v3-small | Google Colab (T4 or A100) | Saves portable `safetensors` checkpoint and tokenizer to `artifacts/classifier/` |
| Stage 2 Bi-Encoder fine-tuning | Google Colab (T4 or A100) | Contrastive `MultipleNegativesRankingLoss` + fast GPU index rebuild |
| Stage 3 Qwen2.5-3B-Instruct LoRA/QLoRA | Google Colab (T4 or A100) | Saves LoRA adapter, merged FP16 weights, and Ollama `Modelfile` |
| End-to-end inference, CLI, and Streamlit UI | Local Laptop (CPU + Ollama) | Uses `artifacts/` checkpoints and GGUF model served by Ollama |

## Repository Structure

| Path | Purpose |
| :--- | :--- |
| `configs/default.yaml` | Central configuration for paths, splits, taxonomy, intent-to-doc mappings, and model hyperparameters |
| `data/org_docs/organization_docs.csv` | 42 tracked organizational policy documents (`DOC001` to `DOC042`) across 12 categories |
| `prompts/` | Jinja2 system (`system.j2`) and user (`user_draft.j2`) prompt templates |
| `src/draft/schemas.py` | Typed Pydantic v2 contracts shared across all three stages |
| `src/draft/data/` | Dataset download, cleaning, canonical stratified splitting, and EDA summary |
| `src/draft/classifier/` | Shared protocol, TF-IDF baseline, and Multi-Task Transformer classifier/trainer |
| `src/draft/retrieval/` | Hybrid FAISS + BM25 store, BGE encoder wrapper, Cross-Encoder reranker, retrieval metrics, contrastive trainer |
| `src/draft/generation/` | Prompt builder, reply sanitizer, citation extractor, Ollama/HF/OpenAI/Mock backends, SFT dataset builder |
| `src/draft/pipeline.py` | Unified `DraftPipeline` orchestrator with per-stage timing |
| `src/draft/evaluation/` | ROUGE, BERTScore, policy adherence, 12-scenario policy suite, and ablation evaluator |
| `src/draft/cli.py` | Typer CLI (`draft`) covering data prep, indexing, retrieval, SFT building, inference, and evaluation |
| `scripts/colab/` | Standalone GPU training and 3-way ablation scripts for Colab (`train_classifier.py`, `train_retriever.py`, `train_llm.py`, `eval_ablation.py`) |
| `notebooks/colab_end_to_end.ipynb` | One-click Colab notebook running all training, GGUF export, and ablation stages with automatic Google Drive backups |
| `app/streamlit_app.py` | Interactive Streamlit UI inspecting all three stages side by side |
| `tests/` | Offline unit and integration tests covering every module |

## Setup with `uv`

Install core dependencies, CPU PyTorch inference stack, Streamlit UI, evaluation metrics, and dev tools locally:

```powershell
uv sync --extra inference --extra ui --extra eval --extra dev
```

For GPU training environments (such as Google Colab):

```bash
pip install -e ".[train,eval]"
```

## Local CLI Workflow

```powershell
# 1. Acquire, clean, and create canonical stratified splits + EDA report
uv run draft prepare-data

# 2. Train and evaluate Stage 1 TF-IDF baseline locally
uv run draft train-baseline

# 3. Build Stage 2 dual FAISS + BM25 retrieval index
uv run draft build-index

# 4. Evaluate Stage 2 retrieval ablations (BM25 vs Dense vs Hybrid vs Boosted vs Reranked)
uv run draft eval-retrieval --split val --max-queries 300 --rerank

# 5. Build leakage-free RAG SFT datasets for Stage 3 fine-tuning
uv run draft build-sft-data --num-ticket-samples 3000 --no-rerank

# 6. Run end-to-end draft generation (mock or ollama backend)
uv run draft generate-draft `
  --subject "Duplicate charge on monthly invoice" `
  --body "We noticed a duplicate charge on our monthly invoice and would like a refund." `
  --generator-backend mock `
  --rerank

# 7. Evaluate end-to-end pipeline and RAG ablation
uv run draft eval-pipeline --split test --max-samples 50 --generator-backend mock --ablation
```

## Colab GPU Training and Local Ollama Deployment

For one-click execution across all stages with automatic Google Drive saving and GPU auto-detection (A100, L4, or T4), upload your source folder (or zip archive) to `MyDrive/draft-rag/`, open `notebooks/colab_end_to_end.ipynb` in Colab, and select **Runtime -> Run all**.

Alternatively, run the standalone GPU scripts individually (A100 recommended; omit `--no-4bit` and lower batch sizes on T4):

```bash
# Stage 1 Multi-Task Transformer
python scripts/colab/train_classifier.py --config configs/default.yaml --output-dir artifacts/classifier --batch-size 64

# Stage 2 Contrastive Bi-Encoder + GPU index rebuild
python scripts/colab/train_retriever.py --config configs/default.yaml --output-dir artifacts/retriever/encoder --batch-size 128 --epochs 2 --rebuild-index

# Refresh SFT JSONL using the fine-tuned Stage 2 retriever
draft build-sft-data --num-ticket-samples 3000 --no-rerank

# Stage 3 LoRA Fine-Tuning + FP16 merge (use default 4-bit without --no-4bit on T4)
python scripts/colab/train_llm.py --config configs/default.yaml --sft-dir data/processed/sft --output-dir artifacts/llm/lora_adapter --epochs 2 --batch-size 8 --grad-accum 2 --max-seq-length 1536 --no-4bit --merge-fp16

# Optional 3-way coursework ablation on GPU (base_no_rag vs base_with_rag vs finetuned_with_rag)
python scripts/colab/eval_ablation.py --config configs/default.yaml --split test --max-samples 100 --no-4bit
```

Convert the merged FP16 model to GGUF (`Q8_0`, pure Python via `llama.cpp`) and register it with Ollama locally:

```bash
python llama.cpp/convert_hf_to_gguf.py artifacts/llm/merged_fp16 --outfile artifacts/llm/draft-qwen25-3b-q8_0.gguf --outtype q8_0
ollama create draft-qwen25-3b -f artifacts/llm/Modelfile
```

Copy `artifacts/classifier/`, `artifacts/retriever/`, `artifacts/llm/`, and `reports/` back to the local workspace so `draft` and the Streamlit UI automatically pick up the fine-tuned models.

## Streamlit Demo UI

```powershell
uv run streamlit run app/streamlit_app.py
```

## Verified Local Baseline Benchmarks

### Stage 1 Classical Baseline (`TfidfBaselineClassifier` on `2,370` Test Emails)

| Split | Intent Acc | Intent Macro-F1 | Intent Weighted-F1 | Urgency Acc | Urgency Macro-F1 | Combined Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Validation (`2,370`) | `0.5494` | `0.5653` | `0.5508` | `0.6304` | `0.6211` | `0.5932` |
| Test (`2,370`) | `0.5308` | `0.5439` | `0.5320` | `0.6004` | `0.5870` | `0.5655` |

### Stage 2 Retrieval Ablations (`300` Validation Queries, Pretrained `BAAI/bge-small-en-v1.5`)

| Mode | Org Docs Hit@1 | Org Docs Hit@3 | Org Docs MRR@3 | Ref Emails Intent Hit@1 | Ref Emails Intent MRR@3 | Ref Emails Tag Hit@3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bm25_only` | `0.3967` | `0.7367` | `0.5389` | `0.7933` | `0.8200` | `0.9967` |
| `dense_only` | `0.5467` | `0.7000` | `0.6178` | `0.8367` | `0.8533` | `1.0000` |
| `hybrid` (`0.65 / 0.35`) | `0.4900` | `0.7467` | `0.6000` | `0.8200` | `0.8433` | `1.0000` |
| `hybrid_boosted` (`+0.15` boost) | `0.9300` | `0.9833` | `0.9533` | `0.9033` | `0.9111` | `1.0000` |
| `hybrid_boosted_reranked` | `0.5700` | `0.7933` | `0.6683` | - | - | - |

## Testing and Linting

```powershell
uv run ruff check src tests scripts app
uv run pytest -v
```
