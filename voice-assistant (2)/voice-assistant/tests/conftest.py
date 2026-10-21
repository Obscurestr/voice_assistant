import datetime as dt
from pathlib import Path

import pytest

from voice_assistant.data import load_dataset
from voice_assistant.memory import MemoryStore
from voice_assistant.nlu import IntentClassifier
from voice_assistant.skills import SkillContext

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def sample_df():
    df, _ = load_dataset(DATA / "sample_intents.csv")
    return df


@pytest.fixture(scope="session")
def classifier(sample_df):
    return IntentClassifier(threshold=0.3).fit(sample_df["text"], sample_df["intent"])


class FakeEnv:
    """Records side effects instead of opening browsers or starting threads."""

    def __init__(self):
        self.opened: list[str] = []
        self.scheduled: list[float] = []
        self.weather_payloads: list = []

    def fetch_json(self, url):
        if not self.weather_payloads:
            raise OSError("offline")
        return self.weather_payloads.pop(0)


@pytest.fixture
def env():
    return FakeEnv()


@pytest.fixture
def ctx(env, tmp_path):
    return SkillContext(
        memory=MemoryStore(tmp_path / "memory.json"),
        open_url=env.opened.append,
        now=lambda: dt.datetime(2026, 10, 21, 9, 5),
        fetch_json=env.fetch_json,
        schedule=lambda seconds, cb: env.scheduled.append(seconds),
        notify=lambda msg: None,
    )
