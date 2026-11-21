"""Generation quality metrics, policy adherence scoring, and unseen policy scenarios."""

import re
from typing import Any

from draft.generation.prompts import extract_cited_doc_ids
from draft.schemas import DraftResponse, OrgDocument, UrgencyLevel

_URGENCY_CUES = (
    "urgent",
    "urgency",
    "immediate",
    "immediately",
    "priority",
    "prioritiz",
    "asap",
    "escalat",
    "right away",
)

_RAW_POLICY_SCENARIOS: list[tuple[str, str, str, str, list[str]]] = [
    (
        "POL_TEST_01",
        "DOC001",
        "When do monthly statements go out and what are payment terms?",
        "Could you remind our finance team which day of the month invoices arrive and how many "
        "days we have to settle balances before overdue reminders trigger?",
        ["first day of each month", "Net 15", "10 days"],
    ),
    (
        "POL_TEST_02",
        "DOC006",
        "Lost phone with authenticator app",
        "One of our workspace admins misplaced their phone and cannot sign in with 2FA. "
        "We also exhausted backup codes, so how can support reset their MFA?",
        ["Multi-Factor Authentication", "10", "Organization Admin"],
    ),
    (
        "POL_TEST_03",
        "DOC009",
        "Hitting HTTP 429 on REST endpoints",
        "Our integration script is hitting rate limit errors. What are the per-minute API quotas "
        "for standard vs enterprise plans and how should retries work?",
        ["120", "600", "Retry-After"],
    ),
    (
        "POL_TEST_04",
        "DOC014",
        "Uploading a large customer CSV migration",
        "We plan to upload a 400,000-row CSV file. Will it exceed import limits and will it run "
        "synchronously or in the background?",
        ["250 MB", "500,000", "50,000"],
    ),
    (
        "POL_TEST_05",
        "DOC017",
        "Planned weekend maintenance schedule",
        "We need to notify our operations team about your regular maintenance window and how "
        "much advance email warning you provide beforehand.",
        ["Sundays", "01:00", "72 hours"],
    ),
    (
        "POL_TEST_06",
        "DOC021",
        "Account erasure under GDPR",
        "A European customer invoked right-to-erasure. How quickly are production records and "
        "backups purged, and how long are financial audit logs kept?",
        ["14 days", "30 days", "7 years"],
    ),
    (
        "POL_TEST_07",
        "DOC024",
        "Exporting over 100k telemetry rows",
        "Which file formats do dashboard exports support, and how are large exports above "
        "100,000 rows delivered?",
        ["CSV", "Parquet", "48 hours"],
    ),
    (
        "POL_TEST_08",
        "DOC027",
        "Outbound email domain authentication rules",
        "Before launching marketing sends on a new domain, what DNS records are mandatory and "
        "what bounce rate triggers a pause?",
        ["SPF", "14-day", "5 percent"],
    ),
    (
        "POL_TEST_09",
        "DOC031",
        "Evaluating the product with a trial workspace",
        "Do prospects need a credit card for a trial, how long do standard vs guided POC "
        "evaluations last, and is configuration kept after expiry?",
        ["14-day", "30-day", "30 days"],
    ),
    (
        "POL_TEST_10",
        "DOC034",
        "Shipping back defective hardware unit",
        "We need to return faulty hardware for replacement. Do we need authorization before "
        "shipping and how fast do you issue a prepaid label?",
        ["RMA", "1 business day", "10 business days"],
    ),
    (
        "POL_TEST_11",
        "DOC038",
        "Laptop and access setup for new employee",
        "A new engineer starts next week. How many days before start should IT provisioning be "
        "submitted, and what is the SLA for missing onboarding access?",
        ["5 business days", "4-hour"],
    ),
    (
        "POL_TEST_12",
        "DOC041",
        "Requesting a supervisor review on stalled ticket",
        "Our issue has dragged past SLA and we want supervisor review. How do we escalate and "
        "how quickly does a Tier-2 Lead respond?",
        ["Tier-2", "4 business hours"],
    ),
]


def _lcs_rouge_l_f1(prediction: str, reference: str) -> float:
    """Compute token-level ROUGE-L F1 score as a pure-Python fallback."""
    pred_tokens = re.findall(r"[a-z0-9]+", prediction.lower())
    ref_tokens = re.findall(r"[a-z0-9]+", reference.lower())
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(pred_tokens), len(ref_tokens)
    prev = [0] * (n + 1)
    for tok in pred_tokens:
        curr = [0] * (n + 1)
        for j, ref_tok in enumerate(ref_tokens, start=1):
            curr[j] = prev[j - 1] + 1 if tok == ref_tok else max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[n]
    if lcs == 0:
        return 0.0
    prec = lcs / float(m)
    rec = lcs / float(n)
    return 2.0 * prec * rec / (prec + rec)


def compute_rouge_l(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute mean ROUGE-1, ROUGE-2, and ROUGE-L F1 scores across predictions."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length.")
    if not predictions:
        return {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0}

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = [
            scorer.score(ref, pred) for pred, ref in zip(predictions, references, strict=True)
        ]
        n = float(len(scores))
        return {
            "rouge1_f1": sum(s["rouge1"].fmeasure for s in scores) / n,
            "rouge2_f1": sum(s["rouge2"].fmeasure for s in scores) / n,
            "rougeL_f1": sum(s["rougeL"].fmeasure for s in scores) / n,
        }
    except ImportError:
        mean_l = sum(
            _lcs_rouge_l_f1(pred, ref) for pred, ref in zip(predictions, references, strict=True)
        ) / float(len(predictions))
        return {"rouge1_f1": mean_l, "rouge2_f1": 0.0, "rougeL_f1": mean_l}


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "distilbert-base-uncased",
    device: str | None = None,
) -> dict[str, float]:
    """Compute mean BERTScore precision, recall, and F1 over prediction-reference pairs."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length.")
    if not predictions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    from bert_score import score as bert_score_fn

    precision, recall, f1 = bert_score_fn(
        predictions,
        references,
        model_type=model_type,
        lang="en",
        device=device,
        verbose=False,
    )
    return {
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
        "f1": float(f1.mean().item()),
    }


def _policy_terms(text: str) -> set[str]:
    """Extract numeric facts and content keywords (>=5 chars) for overlap checks."""
    lower = text.lower()
    numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", lower))
    words = set(re.findall(r"\b[a-z]{5,}\b", lower))
    return numbers | words


def compute_policy_adherence(drafts: list[DraftResponse]) -> dict[str, float]:
    """Compute citation rate, valid citation precision, lexical overlap, and urgency alignment."""
    if not drafts:
        return {
            "citation_rate": 0.0,
            "valid_citation_precision": 0.0,
            "policy_lexical_overlap": 0.0,
            "urgency_alignment_rate": 0.0,
        }

    citation_hits = 0
    citation_precisions: list[float] = []
    overlaps: list[float] = []
    high_total = 0
    high_aligned = 0

    for draft in drafts:
        if draft.cited_doc_ids:
            citation_hits += 1

        raw_cited = extract_cited_doc_ids(draft.body)
        if raw_cited:
            citation_precisions.append(len(draft.cited_doc_ids) / float(len(raw_cited)))
        else:
            citation_precisions.append(1.0)

        docs = draft.retrieved_context.top_k_docs
        if docs:
            doc_terms = set()
            for doc in docs:
                doc_terms.update(_policy_terms(doc.content))
            draft_terms = _policy_terms(draft.body)
            overlaps.append(len(doc_terms & draft_terms) / float(max(len(doc_terms), 1)))
        else:
            overlaps.append(0.0)

        if draft.classification.urgency == UrgencyLevel.HIGH:
            high_total += 1
            body_lower = draft.body.lower()
            if any(cue in body_lower for cue in _URGENCY_CUES):
                high_aligned += 1

    n = float(len(drafts))
    return {
        "citation_rate": citation_hits / n,
        "valid_citation_precision": sum(citation_precisions) / n,
        "policy_lexical_overlap": sum(overlaps) / n,
        "urgency_alignment_rate": (high_aligned / float(high_total)) if high_total > 0 else 1.0,
    }


def build_policy_eval_suite(org_docs: list[OrgDocument]) -> list[dict[str, Any]]:
    """Build 12 unseen policy adherence test scenarios across organization categories."""
    docs_by_id = {doc.doc_id: doc for doc in org_docs}
    suite: list[dict[str, Any]] = []
    for scenario_id, doc_id, subject, body, expected_facts in _RAW_POLICY_SCENARIOS:
        if doc_id not in docs_by_id:
            continue
        doc = docs_by_id[doc_id]
        suite.append(
            {
                "scenario_id": scenario_id,
                "expected_doc_id": doc_id,
                "expected_category": str(doc.category),
                "subject": subject,
                "body": body,
                "expected_facts": expected_facts,
            }
        )
    return suite
