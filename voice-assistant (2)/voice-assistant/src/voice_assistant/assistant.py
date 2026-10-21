"""Orchestrates one conversational turn: text -> intent -> skill -> response."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from .data import normalize_text
from .nlu import IntentClassifier, Prediction
from .skills import REGISTRY, Response, SkillContext

log = logging.getLogger(__name__)

# Statements that store a fact are handled by exact rules before the classifier:
# they are unambiguous, and the training data has few examples of them.
_FACT_RULES: list[tuple[re.Pattern[str], Callable[[SkillContext, str], str]]] = []


def _fact_rule(pattern: str):
    def register(func: Callable[[SkillContext, str], str]):
        _FACT_RULES.append((re.compile(pattern), func))
        return func

    return register


@_fact_rule(r"^(?:my name is|call me|i am called)\s+([a-z][a-z\s'-]{0,40})$")
def _set_name(ctx: SkillContext, value: str) -> str:
    name = value.strip().title()
    ctx.memory["name"] = name
    return f"Nice to meet you, {name}. I'll remember that."


@_fact_rule(r"^my favou?rite (?:music|genre) is\s+(.+)$")
def _set_music(ctx: SkillContext, value: str) -> str:
    prefs = dict(ctx.memory["preferences"])
    prefs["favorite_music"] = value.strip()
    ctx.memory["preferences"] = prefs
    return f"Got it, you like {value.strip()}."


@_fact_rule(r"^(?:i live in|my city is|my home city is)\s+(.+)$")
def _set_city(ctx: SkillContext, value: str) -> str:
    prefs = dict(ctx.memory["preferences"])
    prefs["home_city"] = value.strip().title()
    ctx.memory["preferences"] = prefs
    return f"I'll use {value.strip().title()} for weather from now on."


@dataclass
class Turn:
    user_text: str
    response: Response
    prediction: Prediction | None  # None when a rule handled the turn


class Assistant:
    def __init__(self, classifier: IntentClassifier, context: SkillContext):
        self.classifier = classifier
        self.context = context
        missing = set(classifier.labels) - set(REGISTRY)
        if missing:
            log.warning("No skill registered for intents: %s", sorted(missing))

    def handle(self, text: str) -> Turn:
        norm = normalize_text(text)
        if not norm:
            return Turn(text, Response("I didn't hear anything. Could you say that again?"), None)

        for pattern, action in _FACT_RULES:
            match = pattern.match(norm)
            if match:
                return Turn(text, Response(action(self.context, match.group(1))), None)

        prediction = self.classifier.predict(norm)
        log.info("intent=%s p=%.2f text=%r", prediction.intent, prediction.confidence, text)

        if not prediction.is_confident:
            guess = prediction.intent.replace("_", " ")
            return Turn(
                text,
                Response(f"Sorry, I'm not sure what you meant. Did you want help with {guess}?"),
                prediction,
            )

        handler = REGISTRY.get(prediction.intent)
        if handler is None:
            return Turn(text, Response("I understood, but I can't do that yet."), prediction)
        try:
            response = handler(text, self.context)
        except Exception:
            log.exception("Skill %s failed", prediction.intent)
            response = Response("Something went wrong while doing that.")
        return Turn(text, response, prediction)
