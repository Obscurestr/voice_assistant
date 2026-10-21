import pandas as pd
import pytest

from voice_assistant.data import leak_free_split, load_dataset, normalize_text, strip_row_ids


def test_normalize_removes_punctuation_and_case():
    assert normalize_text("  What's the Weather, today?! ") == "what's the weather today"


def test_strip_row_ids_only_removes_positional_suffix():
    prompts = pd.Series(["wake me up 0", "set a timer for 5", "set an alarm 2"])
    cleaned, stripped = strip_row_ids(prompts)
    assert list(cleaned) == ["wake me up", "set a timer for 5", "set an alarm"]
    assert stripped == 2


def test_load_dataset_deduplicates_and_drops_conflicts(tmp_path):
    csv = tmp_path / "d.csv"
    pd.DataFrame(
        {
            "prompt": ["set an alarm 0", "set an alarm 1", "hello 2", "hello 3", "play jazz 4"],
            "intent": ["alarm", "alarm", "personality", "routine", "play_music"],
        }
    ).to_csv(csv, index=False)
    df, report = load_dataset(csv)
    assert sorted(df["text"]) == ["play jazz", "set an alarm"]
    assert report.conflicting_prompts == ["hello"]
    assert report.row_ids_stripped == 5


def test_missing_columns_raise(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["x"]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_dataset(csv)


def test_split_has_no_overlap(sample_df):
    train, test = leak_free_split(sample_df)
    assert not set(train["text"]) & set(test["text"])
    assert set(test["intent"]) <= set(train["intent"])
