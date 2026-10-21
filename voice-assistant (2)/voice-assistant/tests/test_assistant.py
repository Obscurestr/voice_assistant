from voice_assistant.assistant import Assistant
from voice_assistant.memory import MemoryStore
from voice_assistant.nlu import IntentClassifier
from voice_assistant.skills import REGISTRY


def test_every_dataset_intent_has_a_skill(sample_df):
    assert set(sample_df["intent"]) <= set(REGISTRY)


def test_model_roundtrip(classifier, tmp_path):
    path = tmp_path / "m.joblib"
    classifier.save(path)
    loaded = IntentClassifier.load(path)
    assert loaded.predict("set a timer for 3 minutes").intent == "set_timer"


def test_low_confidence_asks_for_clarification(classifier, ctx):
    turn = Assistant(classifier, ctx).handle("zxqv blorp")
    assert not turn.prediction.is_confident
    assert "not sure" in turn.response.text


def test_end_to_end_conversation(classifier, ctx, env):
    bot = Assistant(classifier, ctx)
    assert "Sara" in bot.handle("My name is Sara").response.text
    bot.handle("my favourite music is metal")
    assert bot.handle("add milk to my shopping list").response.text == (
        "Added milk to your shopping list."
    )
    recall = bot.handle("what do you remember about me").response.text
    assert "Sara" in recall and "metal" in recall and "milk" in recall

    assert bot.handle("set a timer for 10 minutes").response.text == "Timer set for 10 minutes."
    assert env.scheduled == [600]

    bot.handle("play music")
    assert "metal" in env.opened[-1]


def test_memory_persists_to_disk(tmp_path):
    path = tmp_path / "mem.json"
    store = MemoryStore(path)
    store["name"] = "Sara"
    store.append("shopping_list", "bread")
    reloaded = MemoryStore(path)
    assert reloaded["name"] == "Sara"
    assert reloaded["shopping_list"] == ["bread"]
