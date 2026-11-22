"""Data acquisition service orchestrating download, cleaning, splitting, and EDA."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from draft.config import AppConfig
from draft.data.cleaning import clean_email_dataframe
from draft.data.org_docs import build_default_org_docs, load_org_docs, save_org_docs
from draft.data.splitter import create_canonical_splits
from draft.schemas import OrgDocument

PLACEHOLDER_PATTERN = re.compile(r"<[a-zA-Z0-9_]+>")


class DataAcquisitionService:
    """Downloads, cleans, splits email data and prepares organization policy documents."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def download_raw_emails(self, force: bool = False) -> pd.DataFrame:
        """Download or load cached raw tickets from Hugging Face."""
        raw_path = self.config.paths.raw_dir / "customer_support_tickets.parquet"
        if raw_path.exists() and not force:
            return pd.read_parquet(raw_path)

        raw_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(self.config.data.hf_dataset_id, split="train")
        frame = dataset.to_pandas()
        frame.to_parquet(raw_path, index=False)
        return frame

    def ensure_org_docs(self, overwrite: bool = False) -> list[OrgDocument]:
        """Load existing organization docs or synthesize and persist defaults."""
        docs_path = self.config.paths.org_docs_file
        if docs_path.exists() and not overwrite:
            return load_org_docs(docs_path)

        docs = build_default_org_docs()
        save_org_docs(docs, docs_path)
        return docs

    @staticmethod
    def _length_stats(series: pd.Series) -> dict[str, dict[str, int]]:
        """Compute p50, p90, and max character and word counts for a text series."""
        char_lens = series.str.len()
        word_lens = series.str.split().str.len()
        return {
            "chars": {
                "p50": int(char_lens.quantile(0.50)),
                "p90": int(char_lens.quantile(0.90)),
                "max": int(char_lens.max()),
            },
            "words": {
                "p50": int(word_lens.quantile(0.50)),
                "p90": int(word_lens.quantile(0.90)),
                "max": int(word_lens.max()),
            },
        }

    def compute_eda_summary(
        self,
        splits: dict[str, pd.DataFrame],
        cleaning_stats: dict[str, int],
        org_docs: list[OrgDocument],
    ) -> dict[str, Any]:
        """Compute split counts, label balances, text length quantiles, and placeholder stats."""
        combined = pd.concat(splits.values(), ignore_index=True)
        placeholder_counts: Counter[str] = Counter()
        for answer in combined["answer"]:
            placeholder_counts.update(PLACEHOLDER_PATTERN.findall(answer))

        doc_cat_counts = Counter(str(doc.category) for doc in org_docs)
        return {
            "cleaning_stats": cleaning_stats,
            "split_sizes": {name: len(df) for name, df in splits.items()},
            "intent_counts_by_split": {
                name: df["intent"].value_counts().to_dict() for name, df in splits.items()
            },
            "urgency_counts_by_split": {
                name: df["urgency"].value_counts().to_dict() for name, df in splits.items()
            },
            "length_stats": {
                "email": self._length_stats(combined["email"]),
                "answer": self._length_stats(combined["answer"]),
            },
            "top_answer_placeholders": dict(placeholder_counts.most_common(15)),
            "org_docs_total": len(org_docs),
            "org_docs_by_category": dict(sorted(doc_cat_counts.items())),
        }

    def _save_splits(self, splits: dict[str, pd.DataFrame]) -> dict[str, Path]:
        """Write canonical train, val, and test DataFrames to parquet."""
        out_dir = self.config.paths.processed_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}
        for name, frame in splits.items():
            target = out_dir / f"{name}.parquet"
            frame.to_parquet(target, index=False)
            saved[name] = target
        return saved

    def run(
        self,
        force_download: bool = False,
        overwrite_org_docs: bool = False,
    ) -> dict[str, Any]:
        """Execute full data acquisition, cleaning, splitting, and EDA reporting."""
        raw_df = self.download_raw_emails(force=force_download)
        valid_intents = {str(item) for item in self.config.taxonomy.intents}
        valid_urgencies = {str(item) for item in self.config.taxonomy.urgencies}

        clean_df, cleaning_stats = clean_email_dataframe(
            raw_df,
            self.config.data,
            valid_intents=valid_intents,
            valid_urgencies=valid_urgencies,
        )
        splits = create_canonical_splits(clean_df, self.config.data)
        split_paths = self._save_splits(splits)
        org_docs = self.ensure_org_docs(overwrite=overwrite_org_docs)

        summary = self.compute_eda_summary(splits, cleaning_stats, org_docs)
        summary["output_paths"] = {
            **{f"split_{k}": str(v) for k, v in split_paths.items()},
            "org_docs": str(self.config.paths.org_docs_file),
        }

        reports_dir = self.config.paths.reports_dir
        reports_dir.mkdir(parents=True, exist_ok=True)
        summary_path = reports_dir / "data_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["output_paths"]["eda_summary"] = str(summary_path)
        return summary
