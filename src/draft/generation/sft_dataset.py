"""RAG-aware supervised fine-tuning dataset builder combining tickets and policy examples."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from draft.config import AppConfig
from draft.generation.prompts import PromptBuilder, extract_cited_doc_ids, normalize_target_reply
from draft.retrieval.dual_retriever import DualSourceRetriever
from draft.schemas import (
    ClassificationResult,
    DocCategory,
    IncomingEmail,
    IntentLabel,
    OrgDocument,
    RetrievedContext,
    RetrievedDoc,
    UrgencyLevel,
)

DOC_CATEGORY_TO_INTENT: dict[DocCategory, IntentLabel] = {
    DocCategory.BILLING_AND_PAYMENTS: IntentLabel.BILLING_AND_PAYMENTS,
    DocCategory.ACCOUNT_AND_ACCESS: IntentLabel.IT_SUPPORT,
    DocCategory.INTEGRATIONS: IntentLabel.TECHNICAL_SUPPORT,
    DocCategory.PERFORMANCE: IntentLabel.TECHNICAL_SUPPORT,
    DocCategory.OUTAGES_AND_MAINTENANCE: IntentLabel.SERVICE_OUTAGES_AND_MAINTENANCE,
    DocCategory.SECURITY_AND_COMPLIANCE: IntentLabel.IT_SUPPORT,
    DocCategory.DATA_ANALYTICS: IntentLabel.PRODUCT_SUPPORT,
    DocCategory.DIGITAL_MARKETING: IntentLabel.SALES_AND_PRE_SALES,
    DocCategory.SALES_AND_PRE_SALES: IntentLabel.SALES_AND_PRE_SALES,
    DocCategory.RETURNS_AND_EXCHANGES: IntentLabel.RETURNS_AND_EXCHANGES,
    DocCategory.HUMAN_RESOURCES: IntentLabel.HUMAN_RESOURCES,
    DocCategory.GENERAL_INQUIRY: IntentLabel.GENERAL_INQUIRY,
}


def _ensure_doc_in_top_k(
    doc: OrgDocument,
    retrieved_docs: list[RetrievedDoc],
) -> list[RetrievedDoc]:
    """Ensure the target policy document appears inside retrieved_docs for faithful SFT."""
    if any(item.doc_id == doc.doc_id for item in retrieved_docs):
        return retrieved_docs
    target = RetrievedDoc(
        doc_id=doc.doc_id,
        category=doc.category,
        title=doc.title,
        content=doc.content,
        score=1.0,
    )
    if not retrieved_docs:
        return [target]
    return [target, *retrieved_docs[:-1]]


def _synthetic_doc_pairs(doc: OrgDocument) -> list[tuple[IncomingEmail, UrgencyLevel, str]]:
    """Build two realistic inquiry/reply pairs grounded in an organization policy document."""
    intent_cat = str(doc.category)
    urgent_email = IncomingEmail(
        subject=f"Urgent question regarding {doc.title}",
        body=(
            f"Hello team, we urgently need clarification on your policy for {doc.title} "
            f"under {intent_cat}. What timelines and verification steps apply right now?"
        ),
    )
    urgent_reply = (
        f"Thank you for reaching out regarding {doc.title}, <name>. We understand the urgency "
        f"of your request and are prioritizing it immediately. Per our {intent_cat} policy "
        f"[{doc.doc_id}], {doc.content} Please share your account or ticket reference if "
        "additional review is needed so our team can assist right away."
    )

    standard_email = IncomingEmail(
        subject=f"Policy inquiry: {doc.title}",
        body=(
            f"Could you explain the standard guidelines and procedures for {doc.title} "
            f"in {intent_cat}? We want to make sure our team follows the exact process."
        ),
    )
    standard_reply = (
        f"Hello <name>, thank you for contacting support about {doc.title}. "
        f"Under our official {intent_cat} guidelines [{doc.doc_id}], {doc.content} "
        "Please let us know if you have any follow-up questions."
    )
    return [
        (urgent_email, UrgencyLevel.HIGH, urgent_reply),
        (standard_email, UrgencyLevel.MEDIUM, standard_reply),
    ]


def build_policy_grounded_examples(
    org_docs: list[OrgDocument],
    retriever: DualSourceRetriever,
    prompt_builder: PromptBuilder,
    use_reranker: bool = False,
) -> list[dict[str, Any]]:
    """Generate RAG-formatted policy-grounded SFT examples across all organization docs."""
    candidates: list[tuple[OrgDocument, int, IncomingEmail, IntentLabel, UrgencyLevel, str]] = []
    for doc in org_docs:
        intent = DOC_CATEGORY_TO_INTENT[doc.category]
        for variant_idx, (email, urgency, target) in enumerate(_synthetic_doc_pairs(doc), start=1):
            candidates.append((doc, variant_idx, email, intent, urgency, target))

    query_embs = retriever.encoder.encode(
        [item[2].formatted_text for item in candidates],
        is_query=True,
    )
    examples: list[dict[str, Any]] = []
    for idx, (doc, variant_idx, email, intent, urgency, target) in enumerate(candidates):
        docs = retriever._retrieve_org_docs_with_embedding(
            query_text=email.formatted_text,
            query_embedding=query_embs[idx],
            intent=intent,
            use_reranker=use_reranker,
        )
        docs = _ensure_doc_in_top_k(doc, docs)
        responses = retriever._retrieve_ref_responses_with_embedding(
            query_text=email.formatted_text,
            query_embedding=query_embs[idx],
            intent=intent,
        )
        clf = ClassificationResult(
            intent=intent,
            urgency=urgency,
            intent_confidence=1.0,
            urgency_confidence=1.0,
        )
        prompt = prompt_builder.render(
            email,
            clf,
            RetrievedContext(top_k_docs=docs, top_k_responses=responses),
            use_few_shot=False,
        )
        examples.append(
            {
                "example_id": f"POL_{doc.doc_id}_V{variant_idx}",
                "source": "policy_grounded",
                "intent": str(intent),
                "urgency": str(urgency),
                "messages": prompt.to_chat_messages(assistant=normalize_target_reply(target)),
                "cited_doc_ids": extract_cited_doc_ids(target, {d.doc_id for d in docs}),
            }
        )
    return examples


class SftDatasetBuilder:
    """Constructs leakage-free RAG SFT datasets using train tickets and policy docs."""

    def __init__(
        self,
        config: AppConfig,
        retriever: DualSourceRetriever,
        prompt_builder: PromptBuilder,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.prompt_builder = prompt_builder

    def _stratified_sample(self, train_df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
        """Proportionally sample training tickets across intents."""
        if num_samples >= len(train_df):
            return train_df.reset_index(drop=True)
        seed = self.config.data.random_seed
        if num_samples < train_df["intent"].nunique():
            return train_df.sample(n=num_samples, random_state=seed).reset_index(drop=True)
        sampled, _ = train_test_split(
            train_df,
            train_size=num_samples,
            stratify=train_df["intent"],
            random_state=seed,
        )
        return sampled.reset_index(drop=True)

    def _build_ticket_examples(
        self,
        sampled_df: pd.DataFrame,
        use_reranker: bool,
    ) -> list[dict[str, Any]]:
        """Render RAG SFT examples from historical tickets with strict self-exclusion."""
        queries = sampled_df["email"].astype(str).tolist()
        query_embs = self.retriever.encoder.encode(queries, is_query=True)

        examples: list[dict[str, Any]] = []
        for idx, row in enumerate(sampled_df.to_dict(orient="records")):
            email_id = str(row["email_id"])
            intent = IntentLabel(row["intent"])
            urgency = UrgencyLevel(row["urgency"])
            incoming = IncomingEmail(
                subject=str(row.get("subject", "")),
                body=str(row.get("body", row["email"])),
            )
            query_text = queries[idx]
            docs = self.retriever._retrieve_org_docs_with_embedding(
                query_text=query_text,
                query_embedding=query_embs[idx],
                intent=intent,
                use_reranker=use_reranker,
            )
            responses = self.retriever._retrieve_ref_responses_with_embedding(
                query_text=query_text,
                query_embedding=query_embs[idx],
                intent=intent,
                exclude_email_ids={email_id},
            )
            if any(resp.email_id == email_id for resp in responses):
                raise AssertionError(f"Self-leakage detected in SFT context for {email_id}")

            clf = ClassificationResult(
                intent=intent,
                urgency=urgency,
                intent_confidence=1.0,
                urgency_confidence=1.0,
            )
            prompt = self.prompt_builder.render(
                incoming,
                clf,
                RetrievedContext(top_k_docs=docs, top_k_responses=responses),
                use_few_shot=False,
            )
            target = normalize_target_reply(str(row["answer"]))
            examples.append(
                {
                    "example_id": email_id,
                    "source": "ticket",
                    "intent": str(intent),
                    "urgency": str(urgency),
                    "messages": prompt.to_chat_messages(assistant=target),
                    "cited_doc_ids": [],
                }
            )
        return examples

    @staticmethod
    def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
        """Write list of dictionaries to a UTF-8 JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(json.dumps(item, ensure_ascii=False) for item in records)
        path.write_text(content + ("\n" if records else ""), encoding="utf-8")

    @staticmethod
    def _char_stats(lengths: list[int]) -> dict[str, int]:
        """Compute min, median, p90, and max character length statistics."""
        if not lengths:
            return {"min": 0, "p50": 0, "p90": 0, "max": 0}
        series = pd.Series(lengths)
        return {
            "min": int(series.min()),
            "p50": int(series.quantile(0.50)),
            "p90": int(series.quantile(0.90)),
            "max": int(series.max()),
        }

    def build(
        self,
        train_df: pd.DataFrame,
        org_docs: list[OrgDocument],
        output_dir: Path | None = None,
        num_ticket_samples: int = 3000,
        include_policy_grounded: bool = True,
        policy_upsample: int = 3,
        eval_fraction: float = 0.05,
        use_reranker: bool = False,
    ) -> dict[str, Any]:
        """Build, split, and persist RAG-aware SFT JSONL datasets and summary metadata."""
        target_dir = output_dir or (self.config.paths.processed_dir / "sft")
        sampled_df = self._stratified_sample(train_df, num_ticket_samples)
        ticket_examples = self._build_ticket_examples(sampled_df, use_reranker=use_reranker)

        policy_examples: list[dict[str, Any]] = []
        if include_policy_grounded and org_docs:
            base_policy = build_policy_grounded_examples(
                org_docs,
                self.retriever,
                self.prompt_builder,
                use_reranker=use_reranker,
            )
            for copy_idx in range(max(1, policy_upsample)):
                for item in base_policy:
                    policy_examples.append(
                        {
                            **item,
                            "example_id": f"{item['example_id']}_R{copy_idx + 1}",
                        }
                    )

        all_examples = ticket_examples + policy_examples
        sources = [ex["source"] for ex in all_examples]
        has_min_strata = len(set(sources)) > 1 and min(sources.count(s) for s in set(sources)) >= 2
        stratify_arg = sources if has_min_strata else None
        train_records, eval_records = train_test_split(
            all_examples,
            test_size=eval_fraction,
            random_state=self.config.data.random_seed,
            stratify=stratify_arg,
        )

        train_path = target_dir / "sft_train.jsonl"
        eval_path = target_dir / "sft_eval.jsonl"
        self._write_jsonl(train_records, train_path)
        self._write_jsonl(eval_records, eval_path)

        prompt_lens = [
            len(ex["messages"][0]["content"]) + len(ex["messages"][1]["content"])
            for ex in all_examples
        ]
        completion_lens = [len(ex["messages"][2]["content"]) for ex in all_examples]
        summary = {
            "total_examples": len(all_examples),
            "train_examples": len(train_records),
            "eval_examples": len(eval_records),
            "ticket_examples": len(ticket_examples),
            "policy_grounded_examples": len(policy_examples),
            "policy_upsample": policy_upsample if include_policy_grounded else 0,
            "zero_self_leakage": True,
            "prompt_chars": self._char_stats(prompt_lens),
            "completion_chars": self._char_stats(completion_lens),
            "output_paths": {
                "train": str(train_path),
                "eval": str(eval_path),
            },
        }
        summary_path = target_dir / "sft_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["output_paths"]["summary"] = str(summary_path)
        return summary
