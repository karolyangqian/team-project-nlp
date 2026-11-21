"""Canonical stratified dataset splitting with zero-leakage assertions."""

import pandas as pd
from sklearn.model_selection import train_test_split

from draft.config import DataConfig
from draft.data.cleaning import normalize_key


def _pick_stratify_series(df: pd.DataFrame, min_count: int) -> pd.Series | None:
    """Build composite (intent, urgency) stratification labels with rare-stratum fallback."""
    composite = df["intent"].astype(str) + "__" + df["urgency"].astype(str)
    counts = composite.value_counts()
    if counts.min() >= min_count:
        return composite

    # Fallback rare composite pairs to intent-level so stratification remains valid.
    rare_composite = set(counts[counts < min_count].index)
    relaxed = composite.where(~composite.isin(rare_composite), df["intent"].astype(str))
    relaxed_counts = relaxed.value_counts()
    if relaxed_counts.min() >= min_count:
        return relaxed

    intent_only = df["intent"].astype(str)
    if intent_only.value_counts().min() >= min_count:
        return intent_only
    return None


def _assert_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Verify email IDs and normalized texts are strictly disjoint across splits."""
    names = list(splits.keys())
    id_sets = {name: set(df["email_id"]) for name, df in splits.items()}
    text_sets = {name: set(df["email"].apply(normalize_key)) for name, df in splits.items()}

    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared_ids = id_sets[left] & id_sets[right]
            if shared_ids:
                raise AssertionError(
                    f"Email ID leakage between {left} and {right}: {len(shared_ids)}"
                )
            shared_texts = text_sets[left] & text_sets[right]
            if shared_texts:
                raise AssertionError(
                    f"Normalized email text leakage between {left} and {right}: {len(shared_texts)}"
                )


def create_canonical_splits(df: pd.DataFrame, config: DataConfig) -> dict[str, pd.DataFrame]:
    """Create canonical stratified train, val, and test splits."""
    if len(df) < 3:
        raise ValueError("Need at least 3 rows to create train/val/test splits.")

    strat_all = _pick_stratify_series(df, min_count=3)
    train_df, temp_df = train_test_split(
        df,
        test_size=config.val_ratio + config.test_ratio,
        random_state=config.random_seed,
        stratify=strat_all,
    )

    relative_test_size = config.test_ratio / (config.val_ratio + config.test_ratio)
    strat_temp = _pick_stratify_series(temp_df, min_count=2)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=config.random_seed,
        stratify=strat_temp,
    )

    splits = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
    _assert_no_leakage(splits)
    return splits
