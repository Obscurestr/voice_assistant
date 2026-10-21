"""Loading, cleaning and splitting the intent dataset.

The original dataset appends the row index to every prompt ("set an alarm 2",
"set an alarm 3", ...). Those suffixes make otherwise identical sentences look
unique, so a random train/test split puts the same sentence on both sides and
the reported accuracy measures memorisation rather than generalisation.
This module strips the row ids, removes the resulting duplicates and builds a
split in which no sentence can appear in both train and test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {"prompt", "intent"}
_NON_TEXT = re.compile(r"[^a-z0-9:'\s]")
_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation (Whisper adds it) and collapse whitespace."""
    text = _NON_TEXT.sub(" ", str(text).lower())
    return _SPACES.sub(" ", text).strip()


def strip_row_ids(prompts: pd.Series) -> tuple[pd.Series, int]:
    """Remove a trailing number only when it equals that row's position in the file.

    This is deliberately narrow: "set a timer for 5" keeps its 5 unless it happens
    to sit on row 5, so genuine numbers in commands are preserved.
    """
    stripped = 0
    cleaned = []
    for position, prompt in enumerate(prompts.astype(str)):
        suffix = f" {position}"
        if prompt.rstrip().endswith(suffix):
            cleaned.append(prompt.rstrip()[: -len(suffix)])
            stripped += 1
        else:
            cleaned.append(prompt)
    return pd.Series(cleaned, index=prompts.index), stripped


@dataclass
class DatasetReport:
    rows: int
    unique_raw_prompts: int
    row_ids_stripped: int
    unique_clean_prompts: int
    conflicting_prompts: list[str]
    class_counts: dict[str, int] = field(default_factory=dict)

    @property
    def duplicate_rate(self) -> float:
        return 1 - self.unique_clean_prompts / self.rows if self.rows else 0.0


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, DatasetReport]:
    """Read the CSV and return (clean, de-duplicated frame, report on what was removed)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Put your CSV there or set VA_DATA.")
    raw = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    raw = raw.dropna(subset=["prompt", "intent"]).reset_index(drop=True)
    no_ids, stripped = strip_row_ids(raw["prompt"])
    df = pd.DataFrame(
        {
            "raw_prompt": raw["prompt"].astype(str),
            "text": no_ids.map(normalize_text),
            "intent": raw["intent"].astype(str).str.strip(),
        }
    )
    df = df[df["text"] != ""]

    # A sentence labelled with two different intents cannot be learned; report it.
    labels_per_text = df.groupby("text")["intent"].nunique()
    conflicts = sorted(labels_per_text[labels_per_text > 1].index)

    clean = (
        df[~df["text"].isin(conflicts)]
        .drop_duplicates(subset=["text", "intent"])
        .reset_index(drop=True)
    )
    report = DatasetReport(
        rows=len(raw),
        unique_raw_prompts=raw["prompt"].nunique(),
        row_ids_stripped=stripped,
        unique_clean_prompts=len(clean),
        conflicting_prompts=conflicts,
        class_counts=clean["intent"].value_counts().sort_index().to_dict(),
    )
    return clean, report


def leak_free_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split of de-duplicated data, so every test sentence is unseen.

    Intents with a single example cannot be stratified; they go to training.
    """
    counts = df["intent"].value_counts()
    rare = df[df["intent"].isin(counts[counts < 2].index)]
    common = df.drop(rare.index)
    train, test = train_test_split(
        common, test_size=test_size, random_state=seed, stratify=common["intent"]
    )
    train = pd.concat([train, rare])
    overlap = set(train["text"]) & set(test["text"])
    assert not overlap, f"Leakage: {len(overlap)} sentences in both splits"
    return train.reset_index(drop=True), test.reset_index(drop=True)
