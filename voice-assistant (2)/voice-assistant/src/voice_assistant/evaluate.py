"""Model evaluation that measures generalisation, not memorisation.

Produces `reports/results.md`, `reports/metrics.json` and a confusion matrix.
Three numbers are reported, from least to most trustworthy:

1. Naive split   - the original notebook's protocol (random split of raw rows).
                   Kept only to show how much leakage inflates the score.
2. Clean split   - row ids stripped, duplicates removed, stratified split.
3. Held-out set  - hand-written phrasings that never appeared in the dataset.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split  # noqa: E402

from .data import leak_free_split, load_dataset  # noqa: E402
from .nlu import DEFAULT_MODEL, IntentClassifier, candidate_models  # noqa: E402


def naive_split_score(raw_path: Path, seed: int = 42) -> float:
    """Reproduce the notebook: TF-IDF on raw prompts (with ids), random split."""
    raw = pd.read_csv(raw_path).dropna(subset=["prompt", "intent"])
    model = candidate_models(seed)[DEFAULT_MODEL]
    x_tr, x_te, y_tr, y_te = train_test_split(
        raw["prompt"], raw["intent"], test_size=0.2, random_state=seed, stratify=raw["intent"]
    )
    model.fit(x_tr, y_tr)
    return accuracy_score(y_te, model.predict(x_te))


def compare_models(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Stratified k-fold cross-validation on de-duplicated data."""
    n_splits = int(min(5, df["intent"].value_counts().min()))
    cv = StratifiedKFold(n_splits=max(n_splits, 2), shuffle=True, random_state=seed)
    rows = []
    for name, model in candidate_models(seed).items():
        scores = cross_validate(
            model, df["text"], df["intent"], cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=-1
        )
        rows.append(
            {
                "model": name,
                "accuracy": scores["test_accuracy"].mean(),
                "accuracy_std": scores["test_accuracy"].std(),
                "macro_f1": scores["test_f1_macro"].mean(),
                "macro_f1_std": scores["test_f1_macro"].std(),
                "fit_seconds": scores["fit_time"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def plot_confusion(y_true, y_pred, labels: list[str], path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    size = max(6, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("True intent")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Share of true intent (recall)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_evaluation(
    data_path: Path,
    heldout_path: Path | None,
    reports_dir: Path,
    threshold: float = 0.35,
    seed: int = 42,
) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    df, report = load_dataset(data_path)
    metrics: dict = {"dataset": asdict(report) | {"duplicate_rate": report.duplicate_rate}}

    metrics["naive_split_accuracy"] = naive_split_score(data_path, seed)

    comparison = compare_models(df, seed)
    metrics["model_comparison"] = comparison.to_dict(orient="records")

    train, test = leak_free_split(df, seed=seed)
    clf = IntentClassifier(clone(candidate_models(seed)[DEFAULT_MODEL]), threshold).fit(
        train["text"], train["intent"]
    )
    preds = clf.pipeline.predict(test["text"])
    metrics["clean_split"] = {
        "accuracy": accuracy_score(test["intent"], preds),
        "macro_f1": f1_score(test["intent"], preds, average="macro"),
        "n_test": len(test),
    }
    clean_report = classification_report(test["intent"], preds, zero_division=0)
    plot_confusion(
        test["intent"],
        preds,
        clf.labels,
        reports_dir / "confusion_clean_split.png",
        "Clean split: normalised confusion matrix",
    )

    heldout_section = "_No held-out file found._"
    if heldout_path and heldout_path.exists():
        # Final model is trained on all clean data, then scored on unseen phrasings.
        full = IntentClassifier(clone(candidate_models(seed)[DEFAULT_MODEL]), threshold).fit(
            df["text"], df["intent"]
        )
        held = pd.read_csv(heldout_path)
        held = held[held["intent"].isin(full.labels)]
        results = [full.predict(t) for t in held["prompt"]]
        held = held.assign(
            predicted=[r.intent for r in results], confidence=[r.confidence for r in results]
        )
        correct = held["predicted"] == held["intent"]
        accepted = held["confidence"] >= threshold
        metrics["heldout"] = {
            "n": len(held),
            "accuracy": float(correct.mean()),
            "macro_f1": f1_score(held["intent"], held["predicted"], average="macro"),
            "coverage_at_threshold": float(accepted.mean()),
            "accuracy_when_confident": float(correct[accepted].mean()) if accepted.any() else 0.0,
        }
        errors = held[~correct][["prompt", "intent", "predicted", "confidence"]]
        errors.to_csv(reports_dir / "heldout_errors.csv", index=False)
        plot_confusion(
            held["intent"],
            held["predicted"],
            full.labels,
            reports_dir / "confusion_heldout.png",
            "Held-out phrasings: normalised confusion matrix",
        )
        h = metrics["heldout"]
        heldout_section = (
            f"- Utterances: **{h['n']}**, written by hand, none appear in the training data\n"
            f"- Accuracy: **{h['accuracy']:.1%}**, macro F1: **{h['macro_f1']:.3f}**\n"
            f"- At confidence threshold {threshold}: the assistant acts on "
            f"**{h['coverage_at_threshold']:.0%}** of requests, and is right "
            f"**{h['accuracy_when_confident']:.1%}** of the time when it does; "
            f"the rest get a clarifying question.\n"
            f"- Misclassifications: `heldout_errors.csv` ({len(errors)} rows)\n\n"
            "![Held-out confusion matrix](confusion_heldout.png)"
        )

    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    ds = metrics["dataset"]
    n_folds = max(2, int(min(5, min(ds["class_counts"].values()))))
    cs = metrics["clean_split"]
    md = f"""# Evaluation results

Generated by `va evaluate` from `{data_path}`.

## Dataset audit

| | |
|---|---|
| Rows in CSV | {ds["rows"]} |
| Row-id suffixes stripped | {ds["row_ids_stripped"]} |
| Unique sentences after cleaning | {ds["unique_clean_prompts"]} |
| Duplicate rate | {ds["duplicate_rate"]:.1%} |
| Sentences with conflicting labels (removed) | {len(ds["conflicting_prompts"])} |
| Intents | {len(ds["class_counts"])} |

## 1. Naive split (original protocol, leaky)

Accuracy **{metrics["naive_split_accuracy"]:.1%}**. Identical sentences appear in
both train and test, so this mostly measures memorisation.

## 2. Model comparison: {n_folds}-fold CV on de-duplicated data

{_table(comparison.round(3))}

Selected model: **{DEFAULT_MODEL}**. It is close to the best score, fast, and gives
calibrated probabilities, which the assistant needs for its confidence threshold.

Clean hold-out split: accuracy **{cs["accuracy"]:.1%}**, macro F1 **{cs["macro_f1"]:.3f}**
on {cs["n_test"]} unseen sentences.

![Clean split confusion matrix](confusion_clean_split.png)

<details><summary>Per-intent report</summary>

```
{clean_report}
```
</details>

## 3. Held-out phrasings (closest to real use)

{heldout_section}
"""
    (reports_dir / "results.md").write_text(md, encoding="utf-8")
    return metrics
