"""Command-line interface: `va train | evaluate | predict | chat | listen | devices`."""

from __future__ import annotations

import argparse
import logging
import sys

from .assistant import Assistant
from .config import Settings
from .data import load_dataset
from .memory import MemoryStore
from .nlu import IntentClassifier
from .skills import SkillContext


def cmd_train(args, settings: Settings) -> int:
    df, report = load_dataset(settings.data_path)
    print(
        f"Loaded {report.rows} rows -> {report.unique_clean_prompts} unique sentences "
        f"({report.duplicate_rate:.0%} duplicates removed), {len(report.class_counts)} intents."
    )
    clf = IntentClassifier(threshold=settings.confidence_threshold).fit(df["text"], df["intent"])
    clf.save(settings.model_path)
    print(f"Model saved to {settings.model_path}")
    return 0


def cmd_evaluate(args, settings: Settings) -> int:
    from .evaluate import run_evaluation

    metrics = run_evaluation(
        settings.data_path,
        settings.heldout_path,
        settings.reports_dir,
        settings.confidence_threshold,
    )
    print(f"Naive split accuracy (leaky): {metrics['naive_split_accuracy']:.1%}")
    print(f"Clean split accuracy:         {metrics['clean_split']['accuracy']:.1%}")
    if "heldout" in metrics:
        print(f"Held-out phrasing accuracy:   {metrics['heldout']['accuracy']:.1%}")
    print(f"Full report: {settings.reports_dir / 'results.md'}")
    return 0


def _assistant(settings: Settings, notify=print) -> Assistant:
    clf = IntentClassifier.load(settings.model_path, settings.confidence_threshold)
    ctx = SkillContext(
        memory=MemoryStore(settings.memory_path),
        notify=notify,
        documents_dir=settings.documents_dir,
    )
    return Assistant(clf, ctx)


def cmd_predict(args, settings: Settings) -> int:
    clf = IntentClassifier.load(settings.model_path, settings.confidence_threshold)
    p = clf.predict(" ".join(args.text))
    print(f"{p.intent}  ({p.confidence:.2f})" + ("" if p.is_confident else "  [below threshold]"))
    for intent, prob in p.alternatives:
        print(f"  {intent}  ({prob:.2f})")
    return 0


def cmd_chat(args, settings: Settings) -> int:
    assistant = _assistant(settings, notify=lambda msg: print(f"\n[timer] {msg}\n> ", end=""))
    print("Type a command (Ctrl+D or 'quit' to exit).")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text.lower() in {"quit", "exit"}:
            return 0
        turn = assistant.handle(text)
        tag = (
            f"  [{turn.prediction.intent} {turn.prediction.confidence:.2f}]"
            if (args.debug and turn.prediction)
            else ""
        )
        print(turn.response.text + tag)


def cmd_listen(args, settings: Settings) -> int:
    from .audio import Recorder, Speaker, Transcriber, is_silent

    speaker = Speaker(settings.speech_rate)
    assistant = _assistant(settings, notify=speaker.say)
    recorder = Recorder(settings.sample_rate, settings.record_seconds, settings.input_device)
    transcriber = Transcriber(settings.whisper_model)

    name = assistant.context.memory["name"]
    speaker.say(f"Welcome back, {name}." if name else "Hi, I'm ready.")
    while True:
        try:
            cmd = input(f"Press Enter and speak for {settings.record_seconds:g}s (q to quit): ")
        except (EOFError, KeyboardInterrupt):
            return 0
        if cmd.strip().lower() == "q":
            return 0
        audio = recorder.record()
        if is_silent(audio):
            print("(silence)")
            continue
        text = transcriber.transcribe(audio)
        print(f"You: {text}")
        reply = assistant.handle(text).response.text
        print(f"Assistant: {reply}")
        speaker.say(reply)


def cmd_devices(args, settings: Settings) -> int:
    from .audio import Recorder

    print(Recorder.list_devices())
    print("\nSet VA_INPUT_DEVICE=<index> to choose a microphone.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="va", description="Offline voice assistant")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every prediction")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="train the intent model on the dataset")
    sub.add_parser("evaluate", help="audit the data, compare models, write reports/")
    p = sub.add_parser("predict", help="classify one sentence")
    p.add_argument("text", nargs="+")
    c = sub.add_parser("chat", help="text mode, no microphone needed")
    c.add_argument("--debug", action="store_true", help="show intent and confidence")
    sub.add_parser("listen", help="voice mode (needs the [voice] extras)")
    sub.add_parser("devices", help="list audio devices")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    commands = {
        "train": cmd_train, "evaluate": cmd_evaluate, "predict": cmd_predict,
        "chat": cmd_chat, "listen": cmd_listen, "devices": cmd_devices,
    }  # fmt: skip
    try:
        return commands[args.command](args, Settings())
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
