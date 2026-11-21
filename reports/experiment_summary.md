# DRAFT Experimental Results Summary (Presentation Reference)

## 0. Dataset Overview
- Raw tickets: `61,765` -> Deduplicated English: `23,699`
- Stratified splits: Train=`18,959`, Val=`2,370`, Test=`2,370`
- Taxonomy: `10` intents, `3` urgencies, `42` organization policy docs (`12` categories)
- Placeholder frequency in ground-truth replies: `<tel_num>` (9,345), `<name>` (5,510), `<acc_num>` (2,377), `<br>` (581)

## 1. Stage 1: Multi-Task Classification (Test Split, N = 2,370)
| Model | Intent Acc | Intent Macro-F1 | Urgency Acc | Urgency Macro-F1 | Combined Macro-F1 | Latency Mean (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| TF-IDF + LogisticRegression | 0.5308 | 0.5439 | 0.6004 | 0.5870 | 0.5655 | 1.10 |
| DeBERTa-v3-small (4 ep, fp32) | 0.3979 | 0.3950 | 0.5274 | 0.5188 | 0.4569 | 164.13 |

### Stage 1 Top Intent Confusions (TF-IDF Test)
- `Technical Support` -> `Product Support`: **90** (13.2%)
- `Technical Support` -> `IT Support`: **88** (12.9%)
- `Product Support` -> `Technical Support`: **82** (18.6%)
- `Technical Support` -> `Customer Service`: **58** (8.5%)
- `Customer Service` -> `Technical Support`: **58** (16.2%)

## 2. Stage 2: Dual-Source Hybrid RAG Retrieval
### 2A. Organization Policy Docs (Test Split, 300 Queries)
| Mode | Category Hit@1 | Category Hit@3 | Category MRR@3 | Category NDCG@5 |
| :--- | :---: | :---: | :---: | :---: |
| `bm25_only` | 0.4633 | 0.7433 | 0.5833 | 0.6772 |
| `dense_only` | 0.5233 | 0.7900 | 0.6411 | 0.7219 |
| `hybrid` | 0.5600 | 0.7900 | 0.6611 | 0.7236 |
| `hybrid_boosted` | 0.9600 | 0.9933 | 0.9739 | 0.9687 |
| `hybrid_boosted_reranked` | 0.5833 | 0.7933 | 0.6728 | 0.7380 |

### 2B. Contrastive BGE-small Fine-Tuning (Val Split, Hybrid + Boost)
- Category Hit@1: `0.9300` -> `0.9467`
- Category Hit@3: `0.9833` -> `0.9900`
- Category MRR@3: `0.9533` -> `0.9667`

### 2C. Exact Target Policy Doc Hit Rate (12 Unseen Scenarios)
- Baseline + Reranker OFF: Top-1=`83.3%`, Top-3=`91.7%`
- Baseline + Reranker ON: Top-1=`100.0%`, Top-3=`100.0%`
- DeBERTa + Reranker OFF: Top-1=`91.7%`, Top-3=`91.7%`
- DeBERTa + Reranker ON: Top-1=`100.0%`, Top-3=`100.0%`

## 3. Stage 3: Policy-Grounded Draft Generation (5-Way Ablation)
| Mode | Ticket ROUGE-L | Ticket BERTScore F1 | Ticket Leak | Policy Fact Recall | Target Citation Rate | Policy Leak | Stage 3 Mean (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Base Qwen2.5-3B (No RAG) | 0.2108 | 0.8066 | 70.0% | 0.3056 | 0.0000 | 91.7% | 34.6 |
| Base Qwen2.5-3B + RAG (0-Shot) | 0.2930 | 0.8293 | 80.0% | 0.9167 | 0.6667 | 100.0% | 64.7 |
| Base Qwen2.5-3B + RAG + 2-Shot | 0.3981 | 0.8631 | 60.0% | 0.8056 | 0.8333 | 50.0% | 45.8 |
| LoRA SFT + RAG (0-Shot) | 0.6123 | 0.9226 | 60.0% | 0.6944 | 0.2500 | 91.7% | 43.1 |
| LoRA SFT + RAG + 2-Shot | 0.5959 | 0.9137 | 60.0% | 0.2361 | 0.0000 | 75.0% | 44.4 |
