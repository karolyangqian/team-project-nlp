"""Text normalization, tag extraction, and dataset cleaning routines."""

import re

import pandas as pd

from draft.config import DataConfig
from draft.schemas import format_email_text

TAG_COLUMNS = [f"tag_{i}" for i in range(1, 9)]
OUTPUT_COLUMNS = [
    "email_id",
    "subject",
    "body",
    "email",
    "intent",
    "urgency",
    "answer",
    "ticket_type",
    "tags",
]


def clean_text(text: object) -> str:
    """Normalize literal newline escapes, redundant spaces, and blank lines."""
    if text is None or text is pd.NA or pd.isna(text) is True:
        return ""
    value = str(text)
    value = (
        value.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
        .replace("\\t", "\t")
    )
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def extract_tags(row: pd.Series) -> list[str]:
    """Collect non-empty tag_1..tag_8 values from a raw dataset row."""
    tags: list[str] = []
    for col in TAG_COLUMNS:
        if col in row.index:
            val = row[col]
            if val is not None and val is not pd.NA and pd.isna(val) is False:
                tag = str(val).strip()
                if tag:
                    tags.append(tag)
    return tags


def normalize_key(text: str) -> str:
    """Lowercase and collapse whitespace to create a deduplication key."""
    return " ".join(text.lower().split())


def _safe_str_series(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to stripped strings, turning nulls into empty strings."""
    return series.fillna("").astype(str).str.strip()


def clean_email_dataframe(
    df: pd.DataFrame,
    config: DataConfig,
    valid_intents: set[str],
    valid_urgencies: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter, normalize, and deduplicate raw tickets into canonical email records."""
    stats: dict[str, int] = {"raw_rows": len(df)}

    working = df.copy()
    lang_series = _safe_str_series(working["language"]).str.lower()
    working = working[lang_series == config.language.lower()].copy()
    stats["after_language_filter"] = len(working)

    working["intent"] = _safe_str_series(working["queue"])
    working["urgency"] = _safe_str_series(working["priority"]).str.lower()
    working["ticket_type"] = (
        working["type"].apply(clean_text) if "type" in working.columns else ""
    )
    working = working[
        working["intent"].isin(valid_intents) & working["urgency"].isin(valid_urgencies)
    ].copy()
    stats["after_label_filter"] = len(working)

    working["subject"] = working["subject"].apply(clean_text)
    working["body"] = working["body"].apply(clean_text)
    working["answer"] = working["answer"].apply(clean_text)
    working["email"] = [
        format_email_text(subj, body)
        for subj, body in zip(working["subject"], working["body"], strict=True)
    ]

    valid_lengths = (working["email"].str.len() >= config.min_email_chars) & (
        working["answer"].str.len() >= config.min_answer_chars
    )
    working = working[valid_lengths].copy()
    stats["after_text_length_filter"] = len(working)

    working["_dedup_key"] = working["email"].apply(normalize_key)
    working = working.drop_duplicates(subset=["_dedup_key"], keep="first").copy()
    stats["after_deduplication"] = len(working)

    working = working.reset_index(drop=True)
    working["tags"] = working.apply(extract_tags, axis=1)
    working["email_id"] = [f"EML{idx:05d}" for idx in range(1, len(working) + 1)]
    return working[OUTPUT_COLUMNS].copy(), stats
