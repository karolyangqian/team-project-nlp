"""Generate presentation-ready matplotlib figures and a markdown experiment summary."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

PALETTE = {
    "navy": "#1E3A5F",
    "blue": "#2E6FBE",
    "teal": "#1B998B",
    "amber": "#E08E45",
    "coral": "#D94F4F",
    "slate": "#64748B",
    "purple": "#6C5CE7",
    "light_bg": "#F8FAFC",
}

INTENT_SHORT = {
    "Technical Support": "Tech Support",
    "Product Support": "Product Support",
    "Customer Service": "Cust. Service",
    "IT Support": "IT Support",
    "Billing and Payments": "Billing & Pay",
    "Returns and Exchanges": "Returns & Exch",
    "Service Outages and Maintenance": "Outages & Maint",
    "Sales and Pre-Sales": "Sales & Pre",
    "Human Resources": "Human Res.",
    "General Inquiry": "General Inq.",
}


def _configure_style() -> None:
    """Apply crisp academic styling for slide readability."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#E2E8F0",
        }
    )


def _save_fig(fig: plt.Figure, filename: str) -> Path:
    """Save figure at 300 DPI and close cleanly."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[chart] {out_path}")
    return out_path


def _load_json(filename: str) -> dict[str, Any]:
    """Read a JSON report from reports/ using UTF-8."""
    return json.loads((REPORTS_DIR / filename).read_text(encoding="utf-8"))


def plot_dataset_overview(data_summary: dict[str, Any]) -> None:
    """Plot 10-intent split counts and dataset anonymization placeholder frequencies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.4, 1.0]})

    train_counts = data_summary["intent_counts_by_split"]["train"]
    intents = sorted(train_counts.keys(), key=lambda k: train_counts[k])
    short_names = [INTENT_SHORT[k] for k in intents]

    train_vals = [data_summary["intent_counts_by_split"]["train"][k] for k in intents]
    val_vals = [data_summary["intent_counts_by_split"]["val"][k] for k in intents]
    test_vals = [data_summary["intent_counts_by_split"]["test"][k] for k in intents]

    y = np.arange(len(intents))
    ax1.barh(y, train_vals, color=PALETTE["blue"], label="Train (80%)")
    ax1.barh(y, val_vals, left=train_vals, color=PALETTE["teal"], label="Val (10%)")
    left2 = np.array(train_vals) + np.array(val_vals)
    ax1.barh(y, test_vals, left=left2, color=PALETTE["amber"], label="Test (10%)")

    for idx, total in enumerate(left2 + np.array(test_vals)):
        ax1.text(total + 90, idx, f"{int(total):,}", va="center", fontsize=8.5)

    ax1.set_yticks(y)
    ax1.set_yticklabels(short_names)
    ax1.set_xlim(0, 7900)
    ax1.set_xlabel("Number of Deduplicated Tickets (Total = 23,699)")
    ax1.set_title("Dataset Distribution by Intent & Split")
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", visible=False)

    placeholders = data_summary["top_answer_placeholders"]
    top_keys = [k for k in list(placeholders.keys())[:6]]
    top_vals = [placeholders[k] for k in top_keys]
    px = np.arange(len(top_keys))
    bars = ax2.bar(px, top_vals, color=PALETTE["coral"], width=0.58)
    for bar, val in zip(bars, top_vals, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 180,
            f"{val:,}",
            ha="center",
            fontsize=8.5,
        )
    ax2.set_xticks(px)
    ax2.set_xticklabels(top_keys, rotation=20, ha="right")
    ax2.set_ylim(0, max(top_vals) * 1.15)
    ax2.set_ylabel("Occurrences Across Ground-Truth Replies")
    ax2.set_title("Placeholder Tokens in Dataset Replies (SFT Bias Source)")
    ax2.grid(axis="x", visible=False)

    fig.suptitle("DRAFT Dataset Overview: Stratified Splits & Target Reply Artifacts", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "00_dataset_overview.png")


def plot_stage1_charts(
    baseline: dict[str, Any],
    transformer: dict[str, Any],
    stage1_analysis: dict[str, Any],
) -> None:
    """Plot Stage 1 model comparison, per-class F1, and confusion matrices."""
    b_test = baseline["test"]
    t_test = transformer["test"]

    # 01a: Overall metric comparison + latency
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [2.0, 1]}
    )
    metric_labels = [
        "Intent\nAccuracy",
        "Intent\nMacro-F1",
        "Urgency\nAccuracy",
        "Urgency\nMacro-F1",
        "Combined\nMacro-F1",
    ]
    b_scores = [
        b_test["intent"]["accuracy"],
        b_test["intent"]["macro_f1"],
        b_test["urgency"]["accuracy"],
        b_test["urgency"]["macro_f1"],
        b_test["combined_macro_f1"],
    ]
    t_scores = [
        t_test["intent"]["accuracy"],
        t_test["intent"]["macro_f1"],
        t_test["urgency"]["accuracy"],
        t_test["urgency"]["macro_f1"],
        t_test["combined_macro_f1"],
    ]

    x = np.arange(len(metric_labels))
    w = 0.35
    bars1 = ax1.bar(x - w / 2, b_scores, width=w, color=PALETTE["blue"], label="TF-IDF + LogReg")
    bars2 = ax1.bar(
        x + w / 2, t_scores, width=w, color=PALETTE["amber"], label="DeBERTa-v3-small"
    )
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.015,
            f"{h:.3f}",
            ha="center",
            fontsize=8.5,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels)
    ax1.set_ylim(0, 0.72)
    ax1.set_ylabel("Score (Test Split, N = 2,370)")
    ax1.set_title("Stage 1 Classification Performance (Test)")
    ax1.legend(loc="upper right")
    ax1.grid(axis="x", visible=False)

    lat = stage1_analysis["latency_ms"]
    models = ["TF-IDF + LogReg", "DeBERTa-v3-small"]
    mean_ms = [lat["baseline"]["mean_ms"], lat["transformer"]["mean_ms"]]
    p90_ms = [lat["baseline"]["p90_ms"], lat["transformer"]["p90_ms"]]
    lx = np.arange(len(models))
    b_lat1 = ax2.bar(lx - 0.17, mean_ms, width=0.32, color=PALETTE["teal"], label="Mean (ms)")
    b_lat2 = ax2.bar(lx + 0.17, p90_ms, width=0.32, color=PALETTE["navy"], label="P90 (ms)")
    for bar in list(b_lat1) + list(b_lat2):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 5,
            f"{h:.1f}ms",
            ha="center",
            fontsize=8.5,
        )
    ax2.set_xticks(lx)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0, max(p90_ms) * 1.2)
    ax2.set_ylabel("CPU Inference Latency (ms / email)")
    ax2.set_title("Inference Speed Tradeoff (~150x Faster)")
    ax2.legend(loc="upper left")
    ax2.grid(axis="x", visible=False)

    fig.tight_layout()
    _save_fig(fig, "01a_stage1_model_comparison.png")

    # 01b: Per-class Intent F1
    b_per = b_test["intent"]["per_class"]
    t_per = t_test["intent"]["per_class"]
    sorted_intents = sorted(b_per.keys(), key=lambda k: b_per[k]["f1"])
    y = np.arange(len(sorted_intents))
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bh = 0.36
    b_f1 = [b_per[k]["f1"] for k in sorted_intents]
    t_f1 = [t_per[k]["f1"] for k in sorted_intents]
    ax.barh(y + bh / 2, b_f1, height=bh, color=PALETTE["blue"], label="TF-IDF + LogReg")
    ax.barh(y - bh / 2, t_f1, height=bh, color=PALETTE["amber"], label="DeBERTa-v3-small")
    for idx, (bv, tv) in enumerate(zip(b_f1, t_f1, strict=True)):
        ax.text(bv + 0.01, idx + bh / 2, f"{bv:.3f}", va="center", fontsize=8.5)
        ax.text(tv + 0.01, idx - bh / 2, f"{tv:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k} (n={b_per[k]['support']})" for k in sorted_intents])
    ax.set_xlim(0, 0.92)
    ax.set_xlabel("F1 Score on Test Split")
    ax.set_title("Stage 1 Per-Class Intent F1 Breakdown (Keyword-Distinct vs. Overlapping Queues)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save_fig(fig, "01b_stage1_per_class_f1.png")

    # 01c: Confusion matrices (Normalized Intent + Urgency)
    intent_order = list(b_per.keys())
    intent_labels_short = [INTENT_SHORT[k] for k in intent_order]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    for ax, cm_raw, title in (
        (
            axes[0],
            np.array(b_test["intent"]["confusion_matrix"], dtype=float),
            "TF-IDF + LogReg Intent Confusion (Row-Normalized)",
        ),
        (
            axes[1],
            np.array(t_test["intent"]["confusion_matrix"], dtype=float),
            "DeBERTa-v3-small Intent Confusion (Row-Normalized)",
        ),
    ):
        row_norm = cm_raw / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1.0)
        im = ax.imshow(row_norm, cmap="Blues", vmin=0.0, vmax=0.85)
        ax.set_xticks(np.arange(len(intent_labels_short)))
        ax.set_xticklabels(intent_labels_short, rotation=38, ha="right", fontsize=8.5)
        ax.set_yticks(np.arange(len(intent_labels_short)))
        ax.set_yticklabels(intent_labels_short, fontsize=8.5)
        ax.set_xlabel("Predicted Intent")
        ax.set_ylabel("True Intent")
        ax.set_title(title)
        ax.grid(False)
        for i in range(row_norm.shape[0]):
            for j in range(row_norm.shape[1]):
                val = row_norm[i, j]
                if val >= 0.05 or i == j:
                    color = "white" if val > 0.45 else "#0F172A"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    _save_fig(fig, "01c_stage1_confusion_matrices.png")


def plot_stage2_charts(
    ret_test: dict[str, Any],
    ret_train: dict[str, Any],
    stage2_analysis: dict[str, Any],
) -> None:
    """Plot Stage 2 retrieval ablations, BGE fine-tuning gain, and reranker paradox."""
    # 02a: Org docs ablation on test + BGE before/after
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    modes = ["bm25_only", "dense_only", "hybrid", "hybrid_boosted", "hybrid_boosted_reranked"]
    mode_labels = [
        "BM25\nOnly",
        "Dense\n(BGE)",
        "Hybrid\n(RRF)",
        "Hybrid +\nIntent Boost",
        "Hybrid+Boost\n+ Rerank",
    ]
    org = ret_test["org_docs"]
    hit1 = [org[m]["category_hit@1"] for m in modes]
    hit3 = [org[m]["category_hit@3"] for m in modes]
    mrr3 = [org[m]["category_mrr@3"] for m in modes]

    x = np.arange(len(modes))
    w = 0.25
    b1 = ax1.bar(x - w, hit1, width=w, color=PALETTE["blue"], label="Category Hit@1")
    b2 = ax1.bar(x, hit3, width=w, color=PALETTE["teal"], label="Category Hit@3")
    b3 = ax1.bar(x + w, mrr3, width=w, color=PALETTE["navy"], label="Category MRR@3")
    for bar in list(b1) + list(b2) + list(b3):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.015,
            f"{h:.2f}",
            ha="center",
            fontsize=7.5,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(mode_labels)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("Score (Test Split, 300 Queries)")
    ax1.set_title("Stage 2 Org-Doc Retrieval Ablation (Test)")
    ax1.legend(loc="upper left")
    ax1.grid(axis="x", visible=False)

    # Panel 2: Contrastive fine-tuning before vs after (val set from retriever_training_metrics)
    metrics_keys = ["category_hit@1", "category_hit@3", "category_mrr@3", "category_ndcg@5"]
    metric_names = ["Hit@1", "Hit@3", "MRR@3", "NDCG@5"]
    before_vals = [ret_train["before"]["org_docs"][k] for k in metrics_keys]
    after_vals = [ret_train["after"]["org_docs"][k] for k in metrics_keys]
    bx = np.arange(len(metrics_keys))
    bw = 0.34
    bb1 = ax2.bar(
        bx - bw / 2, before_vals, width=bw, color=PALETTE["slate"], label="Pretrained BGE"
    )
    bb2 = ax2.bar(
        bx + bw / 2, after_vals, width=bw, color=PALETTE["teal"], label="Fine-Tuned BGE"
    )
    for bar in list(bb1) + list(bb2):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.01,
            f"{h:.3f}",
            ha="center",
            fontsize=8,
        )
    ax2.set_xticks(bx)
    ax2.set_xticklabels(metric_names)
    ax2.set_ylim(0.80, 1.04)
    ax2.set_ylabel("Score (Hybrid + Boost, Val 300 Queries)")
    ax2.set_title("Contrastive BGE-small Fine-Tuning Impact")
    ax2.legend(loc="lower right")
    ax2.grid(axis="x", visible=False)

    fig.tight_layout()
    _save_fig(fig, "02a_stage2_orgdocs_ablation.png")

    # 02b: Reranker Paradox (Category Hit@1 vs Exact Policy Doc Hit@1 & Hit@3)
    ps = stage2_analysis["policy_suite_retrieval"]
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    categories = [
        "Broad Category\nHit@1 (Test, N=300)",
        "Exact Doc Hit@1\n(Policy, Baseline Clf)",
        "Exact Doc Hit@3\n(Policy, Baseline Clf)",
        "Exact Doc Hit@1\n(Policy, DeBERTa Clf)",
    ]
    off_vals = [
        org["hybrid_boosted"]["category_hit@1"],
        ps["baseline_rerank_off"]["exact_doc_hit@1"],
        ps["baseline_rerank_off"]["exact_doc_hit@3"],
        ps["transformer_rerank_off"]["exact_doc_hit@1"],
    ]
    on_vals = [
        org["hybrid_boosted_reranked"]["category_hit@1"],
        ps["baseline_rerank_on"]["exact_doc_hit@1"],
        ps["baseline_rerank_on"]["exact_doc_hit@3"],
        ps["transformer_rerank_on"]["exact_doc_hit@1"],
    ]
    rx = np.arange(len(categories))
    rw = 0.34
    rb1 = ax.bar(rx - rw / 2, off_vals, width=rw, color=PALETTE["blue"], label="Reranker OFF")
    rb2 = ax.bar(rx + rw / 2, on_vals, width=rw, color=PALETTE["teal"], label="Cross-Encoder ON")
    for bar in list(rb1) + list(rb2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.015,
            f"{h:.1%}",
            ha="center",
            fontsize=8.5,
        )
    ax.set_xticks(rx)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.24)
    ax.set_ylabel("Hit Rate")
    ax.set_title(
        "The Cross-Encoder Paradox: Lower Broad Category Alignment, 100% Exact Policy Rescue"
    )
    ax.legend(loc="upper center", ncol=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save_fig(fig, "02b_stage2_reranker_paradox.png")

    # 02c: Reference email retrieval on test split
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ref_modes = ["bm25_only", "dense_only", "hybrid", "hybrid_boosted"]
    ref_labels = ["BM25 Only", "Dense (BGE)", "Hybrid (RRF)", "Hybrid + Intent Boost"]
    ref = ret_test["ref_emails"]
    r_h1 = [ref[m]["intent_hit@1"] for m in ref_modes]
    r_h3 = [ref[m]["intent_hit@3"] for m in ref_modes]
    r_mrr = [ref[m]["intent_mrr@3"] for m in ref_modes]
    r_tag = [ref[m]["tag_hit@3"] for m in ref_modes]
    tx = np.arange(len(ref_modes))
    tw = 0.2
    for offset, vals, color, lbl in (
        (-1.5 * tw, r_h1, PALETTE["blue"], "Intent Hit@1"),
        (-0.5 * tw, r_h3, PALETTE["teal"], "Intent Hit@3"),
        (0.5 * tw, r_mrr, PALETTE["navy"], "Intent MRR@3"),
        (1.5 * tw, r_tag, PALETTE["amber"], "Tag Hit@3"),
    ):
        bars = ax.bar(tx + offset, vals, width=tw, color=color, label=lbl)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.012,
                f"{h:.2f}",
                ha="center",
                fontsize=7.5,
            )
    ax.set_xticks(tx)
    ax.set_xticklabels(ref_labels)
    ax.set_ylim(0.5, 1.10)
    ax.set_ylabel("Score (Test Split, 300 Queries)")
    ax.set_title("Stage 2 Historical Reference Email Retrieval Ablation (Test)")
    ax.legend(loc="lower right", ncol=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save_fig(fig, "02c_stage2_ref_emails.png")


def plot_stage3_charts(stage3: dict[str, Any]) -> None:
    """Plot Stage 3 5-way ablation, 12-scenario policy heatmap, and pipeline latency."""
    modes = stage3["modes"]
    keys = [
        "qwen25_no_rag",
        "qwen25_rag_0shot",
        "qwen25_rag_2shot",
        "sft_rag_0shot",
        "sft_rag_2shot",
    ]
    short_labels = [
        "Base\nNo-RAG",
        "Base+RAG\n(0-Shot)",
        "Base+RAG\n(2-Shot)",
        "LoRA SFT\n(0-Shot)",
        "LoRA SFT\n(2-Shot)",
    ]

    # 03a: Policy Adherence vs Similarity Tradeoff
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.2))
    policy_recall = [modes[k]["policy_suite"]["policy_fact_recall"] for k in keys]
    cite_rate = [modes[k]["policy_suite"]["target_citation_rate"] for k in keys]
    policy_leak = [modes[k]["policy_suite"]["placeholder_leakage_rate"] for k in keys]

    x = np.arange(len(keys))
    w = 0.25
    for offset, vals, color, lbl in (
        (-w, policy_recall, PALETTE["teal"], "Policy Fact Recall"),
        (0.0, cite_rate, PALETTE["blue"], "Target Citation Rate"),
        (w, policy_leak, PALETTE["coral"], "Placeholder Leak Rate (Lower=Better)"),
    ):
        bars = ax1.bar(x + offset, vals, width=w, color=color, label=lbl)
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.015,
                f"{h:.0%}",
                ha="center",
                fontsize=7.5,
            )
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels)
    ax1.set_ylim(0, 1.18)
    ax1.set_ylabel("Rate (12 Unseen Policy Scenarios)")
    ax1.set_title("Policy Grounding & Reliability Metrics")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(axis="x", visible=False)

    rouge_l = [modes[k]["tickets"]["rouge"]["rougeL_f1"] for k in keys]
    bert_f1 = [modes[k]["tickets"]["bertscore"]["f1"] for k in keys]
    w2 = 0.32
    rb1 = ax2.bar(x - w2 / 2, rouge_l, width=w2, color=PALETTE["amber"], label="ROUGE-L F1")
    rb2 = ax2.bar(x + w2 / 2, bert_f1, width=w2, color=PALETTE["navy"], label="BERTScore F1")
    for bar in list(rb1) + list(rb2):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.015,
            f"{h:.3f}",
            ha="center",
            fontsize=8,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("Score (10 Stratified Test Tickets)")
    ax2.set_title("Surface & Semantic Similarity to Historical Replies")
    ax2.legend(loc="upper left")
    ax2.grid(axis="x", visible=False)

    fig.suptitle(
        "Stage 3 Ablation: Policy Grounding vs. Dataset-Style Mimicry (Qwen2.5-3B vs. LoRA SFT)",
        y=1.02,
    )
    fig.tight_layout()
    _save_fig(fig, "03a_stage3_generation_ablation.png")

    # 03b: 12-Scenario Policy Fact Recall Heatmap
    scenarios = modes[keys[0]]["policy_suite"]["scenarios"]
    scen_labels = [f"{s['scenario_id']} ({s['expected_doc_id']})" for s in scenarios]
    matrix = np.array(
        [[s["fact_recall"] for s in modes[k]["policy_suite"]["scenarios"]] for k in keys],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(13.0, 4.8))
    im = ax.imshow(matrix, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(scen_labels)))
    ax.set_xticklabels(scen_labels, rotation=32, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([lbl.replace("\n", " ") for lbl in short_labels], fontsize=9.5)
    ax.set_title("Policy Fact Recall per Unseen Scenario (12 Categories) Across All 5 Modes")
    ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val >= 0.65 else "#0F172A"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Fact Recall")
    fig.tight_layout()
    _save_fig(fig, "03b_stage3_policy_heatmap.png")

    # 03c: End-to-End Stage Latency Breakdown (seconds)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    s1_s = [modes[k]["tickets"]["latency_ms"]["stage1_ms"]["mean"] / 1000.0 for k in keys]
    s2_s = [modes[k]["tickets"]["latency_ms"]["stage2_ms"]["mean"] / 1000.0 for k in keys]
    s3_s = [modes[k]["tickets"]["latency_ms"]["stage3_ms"]["mean"] / 1000.0 for k in keys]
    p90_tot = [modes[k]["tickets"]["latency_ms"]["total_ms"]["p90"] / 1000.0 for k in keys]

    lx = np.arange(len(keys))
    ax.bar(lx, s1_s, color=PALETTE["teal"], label="Stage 1 (Classify)")
    ax.bar(lx, s2_s, bottom=s1_s, color=PALETTE["blue"], label="Stage 2 (Dual RAG + Rerank)")
    bottom3 = np.array(s1_s) + np.array(s2_s)
    ax.bar(lx, s3_s, bottom=bottom3, color=PALETTE["navy"], label="Stage 3 (Local CPU Ollama)")
    for idx, (mean_tot, p90_val) in enumerate(zip(bottom3 + np.array(s3_s), p90_tot, strict=True)):
        ax.text(
            idx,
            mean_tot + 1.5,
            f"{mean_tot:.1f}s\n(p90 {p90_val:.1f}s)",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(lx)
    ax.set_xticklabels(short_labels)
    ax.set_ylim(0, max(bottom3 + np.array(s3_s)) * 1.2)
    ax.set_ylabel("Mean Latency (Seconds / Email)")
    ax.set_title("End-to-End DRAFT Pipeline Latency Breakdown (Stage 1+2 <= 2.0s vs. Stage 3 CPU)")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save_fig(fig, "03c_pipeline_latency_breakdown.png")


def write_summary_markdown(
    data_summary: dict[str, Any],
    baseline: dict[str, Any],
    transformer: dict[str, Any],
    stage1_analysis: dict[str, Any],
    ret_test: dict[str, Any],
    ret_train: dict[str, Any],
    stage2_analysis: dict[str, Any],
    stage3: dict[str, Any],
) -> None:
    """Write a concise markdown summary of all numerical results and error examples."""
    b_t = baseline["test"]
    t_t = transformer["test"]
    s1_lat = stage1_analysis["latency_ms"]
    ps = stage2_analysis["policy_suite_retrieval"]

    lines = [
        "# DRAFT Experimental Results Summary (Presentation Reference)",
        "",
        "## 0. Dataset Overview",
        f"- Raw tickets: `{data_summary['cleaning_stats']['raw_rows']:,}` -> "
        f"Deduplicated English: `{data_summary['cleaning_stats']['after_deduplication']:,}`",
        f"- Stratified splits: Train=`{data_summary['split_sizes']['train']:,}`, "
        f"Val=`{data_summary['split_sizes']['val']:,}`, "
        f"Test=`{data_summary['split_sizes']['test']:,}`",
        "- Taxonomy: `10` intents, `3` urgencies, `42` organization policy docs (`12` categories)",
        "- Placeholder frequency in ground-truth replies: "
        + ", ".join(
            f"`{k}` ({v:,})"
            for k, v in list(data_summary["top_answer_placeholders"].items())[:4]
        ),
        "",
        "## 1. Stage 1: Multi-Task Classification (Test Split, N = 2,370)",
        "| Model | Intent Acc | Intent Macro-F1 | Urgency Acc | Urgency Macro-F1 | "
        "Combined Macro-F1 | Latency Mean (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| TF-IDF + LogisticRegression | {b_t['intent']['accuracy']:.4f} | "
        f"{b_t['intent']['macro_f1']:.4f} | {b_t['urgency']['accuracy']:.4f} | "
        f"{b_t['urgency']['macro_f1']:.4f} | {b_t['combined_macro_f1']:.4f} | "
        f"{s1_lat['baseline']['mean_ms']:.2f} |",
        f"| DeBERTa-v3-small (4 ep, fp32) | {t_t['intent']['accuracy']:.4f} | "
        f"{t_t['intent']['macro_f1']:.4f} | {t_t['urgency']['accuracy']:.4f} | "
        f"{t_t['urgency']['macro_f1']:.4f} | {t_t['combined_macro_f1']:.4f} | "
        f"{s1_lat['transformer']['mean_ms']:.2f} |",
        "",
        "### Stage 1 Top Intent Confusions (TF-IDF Test)",
    ]
    for err in stage1_analysis["top_intent_confusions"]["baseline"][:5]:
        lines.append(
            f"- `{err['true']}` -> `{err['pred']}`: **{err['count']}** ({err['row_share']:.1%})"
        )

    lines.extend(
        [
            "",
            "## 2. Stage 2: Dual-Source Hybrid RAG Retrieval",
            "### 2A. Organization Policy Docs (Test Split, 300 Queries)",
            "| Mode | Category Hit@1 | Category Hit@3 | Category MRR@3 | Category NDCG@5 |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for mode in ("bm25_only", "dense_only", "hybrid", "hybrid_boosted", "hybrid_boosted_reranked"):
        m = ret_test["org_docs"][mode]
        lines.append(
            f"| `{mode}` | {m['category_hit@1']:.4f} | {m['category_hit@3']:.4f} | "
            f"{m['category_mrr@3']:.4f} | {m['category_ndcg@5']:.4f} |"
        )

    lines.extend(
        [
            "",
            "### 2B. Contrastive BGE-small Fine-Tuning (Val Split, Hybrid + Boost)",
            f"- Category Hit@1: `{ret_train['before']['org_docs']['category_hit@1']:.4f}` -> "
            f"`{ret_train['after']['org_docs']['category_hit@1']:.4f}`",
            f"- Category Hit@3: `{ret_train['before']['org_docs']['category_hit@3']:.4f}` -> "
            f"`{ret_train['after']['org_docs']['category_hit@3']:.4f}`",
            f"- Category MRR@3: `{ret_train['before']['org_docs']['category_mrr@3']:.4f}` -> "
            f"`{ret_train['after']['org_docs']['category_mrr@3']:.4f}`",
            "",
            "### 2C. Exact Target Policy Doc Hit Rate (12 Unseen Scenarios)",
            f"- Baseline + Reranker OFF: Top-1=`{ps['baseline_rerank_off']['exact_doc_hit@1']:.1%}`"
            f", Top-3=`{ps['baseline_rerank_off']['exact_doc_hit@3']:.1%}`",
            f"- Baseline + Reranker ON: Top-1=`{ps['baseline_rerank_on']['exact_doc_hit@1']:.1%}`, "
            f"Top-3=`{ps['baseline_rerank_on']['exact_doc_hit@3']:.1%}`",
            f"- DeBERTa + Reranker OFF: Top-1="
            f"`{ps['transformer_rerank_off']['exact_doc_hit@1']:.1%}`, "
            f"Top-3=`{ps['transformer_rerank_off']['exact_doc_hit@3']:.1%}`",
            f"- DeBERTa + Reranker ON: Top-1="
            f"`{ps['transformer_rerank_on']['exact_doc_hit@1']:.1%}`, "
            f"Top-3=`{ps['transformer_rerank_on']['exact_doc_hit@3']:.1%}`",
            "",
            "## 3. Stage 3: Policy-Grounded Draft Generation (5-Way Ablation)",
            "| Mode | Ticket ROUGE-L | Ticket BERTScore F1 | Ticket Leak | "
            "Policy Fact Recall | Target Citation Rate | Policy Leak | Stage 3 Mean (s) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    stage3_keys = (
        "qwen25_no_rag",
        "qwen25_rag_0shot",
        "qwen25_rag_2shot",
        "sft_rag_0shot",
        "sft_rag_2shot",
    )
    for k in stage3_keys:
        m = stage3["modes"][k]
        t = m["tickets"]
        p = m["policy_suite"]
        lines.append(
            f"| {m['label']} | {t['rouge']['rougeL_f1']:.4f} | {t['bertscore']['f1']:.4f} | "
            f"{t['placeholder_leakage_rate']:.1%} | {p['policy_fact_recall']:.4f} | "
            f"{p['target_citation_rate']:.4f} | {p['placeholder_leakage_rate']:.1%} | "
            f"{t['latency_ms']['stage3_ms']['mean'] / 1000.0:.1f} |"
        )

    out_path = REPORTS_DIR / "experiment_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summary] {out_path}")


def main() -> None:
    """Load all report JSONs, generate figures, and write markdown summary."""
    _configure_style()
    data_summary = _load_json("data_summary.json")
    baseline = _load_json("classifier_baseline_metrics.json")
    transformer = _load_json("classifier_transformer_metrics.json")
    stage1_analysis = _load_json("stage1_analysis.json")
    ret_test = _load_json("retrieval_metrics.json")
    ret_train = _load_json("retriever_training_metrics.json")
    stage2_analysis = _load_json("stage2_analysis.json")
    stage3 = _load_json("stage3_ablation.json")

    plot_dataset_overview(data_summary)
    plot_stage1_charts(baseline, transformer, stage1_analysis)
    plot_stage2_charts(ret_test, ret_train, stage2_analysis)
    plot_stage3_charts(stage3)
    write_summary_markdown(
        data_summary,
        baseline,
        transformer,
        stage1_analysis,
        ret_test,
        ret_train,
        stage2_analysis,
        stage3,
    )


if __name__ == "__main__":
    main()
