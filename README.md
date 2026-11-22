# DRAFT: Sistem RAG Dua Sumber untuk Penyusunan Draf Balasan Email Pelanggan

DRAFT adalah pipeline NLP tiga tahap untuk membantu tim layanan pelanggan menyusun draf balasan email secara otomatis dan terverifikasi. Saat email masuk diterima, sistem memprediksi kategori intent dan tingkat urgensi, mencari dokumen kebijakan perusahaan sekaligus contoh balasan historis yang relevan, lalu menghasilkan draf balasan lengkap dengan sitasi `[DOCxxx]`.

## Arsitektur Pipeline

| Tahap | Direktori | Tugas | Metode & Model |
| :--- | :--- | :--- | :--- |
| Stage 1: Klasifikasi | `src/draft/classifier/` | Klasifikasi 10 kelas intent dan 3 level urgensi (`low`, `medium`, `high`) | TF-IDF + `LogisticRegression` (class-weighted) dan Multi-Task `microsoft/deberta-v3-small` |
| Stage 2: Retrieval Dua Sumber | `src/draft/retrieval/` | Pengambilan top-K dokumen kebijakan (42 dokumen) dan contoh balasan historis (18.959 tiket latih) | `BAAI/bge-small-en-v1.5` (FAISS) + `BM25Okapi` via Reciprocal Rank Fusion + soft intent boost + reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Stage 3: Generasi Draf | `src/draft/generation/` | Penyusunan balasan berbasis kebijakan dengan gaya bahasa tiket historis | Template Jinja2 + 2-shot policy grounding pada `Qwen2.5-3B` (Base Ollama vs. LoRA SFT GGUF) |
| Orkestrasi & Evaluasi | `src/draft/pipeline.py`, `src/draft/evaluation/` | Eksekusi terpadu, pencatatan latensi, dan evaluasi kuantitatif | ROUGE-L, BERTScore, citation rate, policy fact recall (12 skenario), dan placeholder leakage rate |

## Struktur Repositori

| Path | Keterangan |
| :--- | :--- |
| `configs/default.yaml` | Konfigurasi terpusat untuk path, taksonomi label, pemetaan intent-ke-dokumen, dan hyperparameter |
| `data/org_docs/organization_docs.csv` | 42 dokumen kebijakan perusahaan (`DOC001`–`DOC042`) lintas 12 kategori |
| `prompts/` | Template prompt Jinja2 (`system.j2`, `user_draft.j2`) dan contoh 2-shot (`few_shot.json`) |
| `src/draft/` | Paket inti pipeline (`data/`, `classifier/`, `retrieval/`, `generation/`, `evaluation/`, `pipeline.py`, `cli.py`) |
| `app/streamlit_app.py` | Antarmuka web interaktif berbasis Streamlit untuk menguji ketiga tahap secara langsung |
| `scripts/download_artifacts.py` | Utilitas pengunduh bobot model dan indeks siap pakai dari Google Drive |
| `scripts/run_experiments.py` | Skrip reproduksi eksperimen Stage 1–3 dan ablasi 5-mode |
| `scripts/colab/` | Skrip pelatihan dan ablasi GPU mandiri untuk Google Colab |
| `notebooks/colab_end_to_end.ipynb` | Notebook Colab end-to-end untuk pelatihan, ekspor GGUF, dan evaluasi |
| `reports/` | Ringkasan hasil eksperimen (`experiment_summary.md`), JSON metrik, dan grafik (`figures/`) |
| `tests/` | Unit test dan integration test offline (`pytest`) |

## Peringatan Ukuran Unduhan & Kebutuhan Perangkat

Sebelum menjalankan instalasi penuh, perhatikan perkiraan ukuran unduhan dan kebutuhan memori berikut:

| Komponen / Perintah | Perkiraan Ukuran | Catatan Perangkat | Wajib? |
| :--- | :---: | :--- | :---: |
| `uv sync --extra dev` + `uv run pytest` | ~300–400 MB | Berjalan offline tanpa unduh model (~15–30 detik) | Opsional (cek cepat) |
| `uv sync --extra inference --extra ui` | **~1,5–2,5 GB** | Memasang PyTorch CPU, Transformers, FAISS, dan Streamlit | Wajib untuk jalan lokal |
| `download_artifacts.py --profile core` | ~142 MB | Baseline TF-IDF + encoder & indeks retrieval siap pakai | Disarankan |
| `download_artifacts.py --profile no_llm` | ~541 MB | Menambah checkpoint DeBERTa-v3-small (399 MB zip) | Opsional |
| `download_artifacts.py --profile full` | **~3,8 GB** | Menambah artefak GGUF fine-tuned (~3,26 GB) untuk perbandingan LoRA | Opsional |
| `ollama pull qwen2.5:3b` | **~1,9 GB** | Butuh ~4 GB RAM; CPU ~30–60 detik/email (`< 2 detik` di GPU) | Wajib untuk LLM lokal |
| `--extra eval` + `--bertscore` | ~270 MB | Mengunduh `distilbert-base-uncased` untuk BERTScore | Opsional (evaluasi) |
| `--generator-backend hf` | **~6,2 GB** | Mengunduh bobot FP16 `Qwen2.5-3B-Instruct`; hindari di CPU | Tidak disarankan di CPU |

Jika hanya ingin memverifikasi kode tanpa mengunduh model LLM, gunakan `--generator-backend mock`.

## Cara Menjalankan Secara Lokal

### Prasyarat
- Python 3.11+ dan [`uv`](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) (untuk menjalankan model LLM lokal pada Stage 3)

### 1. Instalasi Dependensi

```powershell
uv sync --extra inference --extra ui
```

*(Tambahkan `--extra eval --extra dev` jika ingin menjalankan evaluasi BERTScore, `pytest`, atau `ruff`.)*

### 2. Siapkan Bobot Model & Indeks Retrieval

**Opsi A — Unduh Artefak Siap Pakai dari Google Drive (Disarankan)**

Pilihan paling cepat tanpa perlu melatih atau mengindeks ulang dari awal:

```powershell
# Profil core (~142 MB): TF-IDF baseline + encoder & indeks retrieval siap pakai
uv run --with gdown python scripts/download_artifacts.py --profile core

# Profil no_llm (~541 MB): termasuk checkpoint DeBERTa-v3-small
uv run --with gdown python scripts/download_artifacts.py --profile no_llm

# Profil full (~3,8 GB): seluruh artefak termasuk GGUF fine-tuned
uv run --with gdown python scripts/download_artifacts.py --profile full
```

**Opsi B — Bangun Data & Indeks Sendiri Secara Lokal (Tanpa Google Drive / Colab)**

```powershell
# 1. Unduh & bersihkan dataset tiket (~60-100 MB) lalu buat split 80/10/10
uv run draft prepare-data

# 2. Latih baseline Stage 1 TF-IDF (< 30 detik di CPU)
uv run draft train-baseline

# 3. Bangun indeks FAISS + BM25 (~225 MB unduhan model HF, ~3-6 menit di CPU)
uv run draft build-index
```

### 3. Siapkan Model LLM di Ollama

```powershell
# Model dasar yang direkomendasikan untuk grounded policy drafting (~1,9 GB)
ollama pull qwen2.5:3b

# Opsional: daftarkan model hasil fine-tuning LoRA jika mengunduh --profile full
ollama create draft-qwen25-3b -f artifacts/llm/Modelfile
```

### 4. Jalankan Demo via Command Prompt (CLI)

```powershell
# Konfigurasi rekomendasi (Base Qwen2.5-3B + Dual RAG + Cross-Encoder + 2-Shot)
uv run draft generate-draft `
  --subject "Duplicate charge on invoice" `
  --body "We noticed a duplicate charge on our invoice issued 10 days ago. Are we within the dispute window and where can we download the PDF?" `
  --classifier-backend baseline `
  --generator-backend ollama `
  --ollama-model qwen2.5:3b `
  --few-shot

# Uji cepat tanpa Ollama (menggunakan generator mock deterministik)
uv run draft generate-draft `
  --subject "Duplicate charge on invoice" `
  --body "We noticed a duplicate charge on our invoice issued 10 days ago." `
  --classifier-backend baseline `
  --generator-backend mock
```

Ganti `--ollama-model draft-qwen25-3b` atau tambahkan `--no-few-shot` / `--no-rerank` untuk membandingkan mode secara langsung.

### 5. Jalankan Antarmuka Streamlit (Opsional)

```powershell
uv run streamlit run app/streamlit_app.py
```

## Pelatihan Model di Google Colab

Seluruh proses pelatihan GPU (Stage 1 `DeBERTa-v3-small`, Stage 2 contrastive `bge-small`, Stage 3 QLoRA `Qwen2.5-3B-Instruct`, ekspor GGUF `Q8_0`, dan evaluasi ablasi) dirangkum di `notebooks/colab_end_to_end.ipynb`. Buka notebook di Google Colab dengan runtime GPU (A100/L4/T4), atur `GITHUB_REPO_URL` atau folder Google Drive pada Cell konfigurasi, lalu pilih **Runtime -> Run all**.

## Ringkasan Hasil Eksperimen

Detail lengkap dan grafik tersedia di `reports/experiment_summary.md` serta `reports/figures/`.

### Stage 1: Klasifikasi Intent & Urgensi (Test Split, N = 2.370)

| Model | Intent Acc | Intent Macro-F1 | Urgency Macro-F1 | Combined Macro-F1 | Latensi CPU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| TF-IDF + LogisticRegression | `0.5308` | `0.5439` | `0.5870` | `0.5655` | `1.10 ms` |
| DeBERTa-v3-small (4 epoch, fp32) | `0.3979` | `0.3950` | `0.5188` | `0.4569` | `164.13 ms` |

### Stage 2: Retrieval Dokumen Kebijakan (Test Split 300 Query & 12 Skenario Kebijakan)

| Mode | Category Hit@1 | Category Hit@3 | Category MRR@3 | Exact Policy Hit@1 (12 Skenario) |
| :--- | :---: | :---: | :---: | :---: |
| `bm25_only` | `0.4633` | `0.7433` | `0.5833` | - |
| `dense_only` | `0.5233` | `0.7900` | `0.6411` | - |
| `hybrid` (RRF) | `0.5600` | `0.7900` | `0.6611` | - |
| `hybrid_boosted` | `0.9600` | `0.9933` | `0.9739` | `83.3%` (`10/12`) |
| `hybrid_boosted_reranked` | `0.5833` | `0.7933` | `0.6728` | **`100.0%` (`12/12`)** |

### Stage 3: Ablasi 5-Mode Generasi Draf (10 Tiket Berstrata + 12 Skenario Kebijakan)

| Mode | ROUGE-L | BERTScore F1 | Policy Fact Recall | Target Citation Rate | Policy Leak Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Base `Qwen2.5-3B` (No RAG) | `0.2108` | `0.8066` | `0.3056` | `0.0000` | `91.7%` |
| Base `Qwen2.5-3B` + RAG (0-Shot) | `0.2930` | `0.8293` | **`0.9167`** | `0.6667` | `100.0%` |
| Base `Qwen2.5-3B` + RAG + 2-Shot | `0.3981` | `0.8631` | `0.8056` | **`0.8333`** | **`50.0%`** |
| LoRA SFT + RAG (0-Shot) | **`0.6123`** | **`0.9226`** | `0.6944` | `0.2500` | `91.7%` |
| LoRA SFT + RAG + 2-Shot | `0.5959` | `0.9137` | `0.2361` | `0.0000` | `75.0%` |

## Pengujian & Linting

```powershell
uv run ruff check src tests scripts app
uv run pytest -v
```
