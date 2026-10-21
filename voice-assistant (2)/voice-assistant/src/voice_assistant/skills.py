"""Skills: one handler per intent, registered with a decorator.

Handlers never touch the browser, network or clock directly. They go through a
SkillContext, so every skill can be unit-tested with fakes and the assistant
can run headless.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
import threading
import urllib.parse
import urllib.request
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import entities as ent
from .data import normalize_text
from .documents import open_file, write_shopping_list
from .memory import MemoryStore


@dataclass
class Response:
    text: str
    url: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


def _fetch_json(url: str, timeout: float = 5.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed hosts)
        return json.loads(resp.read().decode("utf-8"))


def _schedule(seconds: float, callback: Callable[[], None]) -> None:
    timer = threading.Timer(seconds, callback)
    timer.daemon = True
    timer.start()


@dataclass
class SkillContext:
    memory: MemoryStore
    open_url: Callable[[str], Any] = webbrowser.open
    now: Callable[[], dt.datetime] = dt.datetime.now
    fetch_json: Callable[[str], Any] = _fetch_json
    schedule: Callable[[float, Callable[[], None]], None] = _schedule
    notify: Callable[[str], None] = print  # how timers announce themselves
    documents_dir: Path = field(
        default_factory=lambda: Path.home() / "Documents" / "Voice Assistant"
    )
    open_file: Callable[[Path], Any] = open_file


Handler = Callable[[str, SkillContext], Response]
REGISTRY: dict[str, Handler] = {}


def skill(*intents: str) -> Callable[[Handler], Handler]:
    def register(func: Handler) -> Handler:
        for intent in intents:
            REGISTRY[intent] = func
        return func

    return register


def _search(ctx: SkillContext, query: str, site: str = "https://www.google.com/search?q=") -> str:
    url = site + urllib.parse.quote_plus(query)
    ctx.open_url(url)
    return url


def _has(text: str, *words: str) -> bool:
    text = normalize_text(text)
    return any(f" {w} " in f" {text} " for w in words)


# --- time & scheduling --------------------------------------------------------------


@skill("date_time")
def tell_time(text: str, ctx: SkillContext) -> Response:
    now = ctx.now()
    if _has(text, "date", "day", "today"):
        return Response(f"Today is {now:%A, %B} {now.day}, {now.year}.")
    return Response(f"It's {ent.format_time(now.time())}.")


@skill("set_timer")
def set_timer(text: str, ctx: SkillContext) -> Response:
    seconds = ent.parse_duration(text)
    if not seconds:
        return Response("How long should the timer be?")
    label = ent.format_duration(seconds)
    ctx.schedule(seconds, lambda: ctx.notify(f"Your {label} timer is done."))
    return Response(f"Timer set for {label}.", data={"seconds": seconds})


@skill("alarm")
def set_alarm(text: str, ctx: SkillContext) -> Response:
    if _has(text, "cancel", "delete", "remove", "turn off"):
        count = len(ctx.memory["alarms"])
        ctx.memory["alarms"] = []
        return Response(f"Removed {count} alarm{'s' if count != 1 else ''}.")
    when = ent.parse_time(text)
    if when is None:
        return Response("What time should I set the alarm for?")
    ctx.memory.append("alarms", when.strftime("%H:%M"))
    return Response(f"Alarm set for {ent.format_time(when)}.", data={"time": when.isoformat()})


@skill("reminder")
def reminder(text: str, ctx: SkillContext) -> Response:
    if _has(text, "what", "show", "list", "my reminders"):
        items = ctx.memory["reminders"]
        if not items:
            return Response("You have no reminders.")
        return Response("Your reminders: " + "; ".join(r["task"] for r in items) + ".")
    task = ent.extract_after(text, ("remind me to", "reminder to", "remind me about", "remind me"))
    if not task:
        return Response("What should I remind you about?")
    when = ent.parse_time(text)
    # keep the task text clean of the time phrase
    task = task.split(" at ")[0].strip()
    ctx.memory.append("reminders", {"task": task, "time": when.strftime("%H:%M") if when else None})
    suffix = f" at {ent.format_time(when)}" if when else ""
    return Response(f"Okay, I'll remind you to {task}{suffix}.")


@skill("calendar")
def calendar(text: str, ctx: SkillContext) -> Response:
    alarms, reminders = ctx.memory["alarms"], ctx.memory["reminders"]
    if not alarms and not reminders:
        return Response("Your schedule is clear. No alarms or reminders saved.")
    parts = []
    if reminders:
        parts.append(f"{len(reminders)} reminder{'s' if len(reminders) > 1 else ''}")
    if alarms:
        parts.append(f"alarms at {', '.join(alarms)}")
    return Response("You have " + " and ".join(parts) + ".")


@skill("routine")
def routine(text: str, ctx: SkillContext) -> Response:
    """'Good morning' routine: greeting, time, and today's agenda."""
    name = ctx.memory["name"]
    hour = ctx.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    greeting += f", {name}." if name else "."
    return Response(
        " ".join([greeting, tell_time("time", ctx).text, calendar("", ctx).text]),
    )


# --- information ----------------------------------------------------------------------

WEATHER_CODES = {
    0: "clear skies", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog",
    51: "light drizzle", 61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow",
    73: "snow", 75: "heavy snow", 80: "rain showers", 95: "a thunderstorm",
}  # fmt: skip


@skill("weather")
def weather(text: str, ctx: SkillContext) -> Response:
    """Live forecast from Open-Meteo (free, no API key); falls back to a web search."""
    city = ent.extract_place(text) or ctx.memory["preferences"].get("home_city")
    if not city:
        return Response("Which city should I check the weather for?")
    try:
        geo = ctx.fetch_json(
            "https://geocoding-api.open-meteo.com/v1/search?count=1&name="
            + urllib.parse.quote(city)
        )
        place = geo["results"][0]
        forecast = ctx.fetch_json(
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={place['latitude']}&longitude={place['longitude']}"
            "&current=temperature_2m,weather_code"
        )
        current = forecast["current"]
        sky = WEATHER_CODES.get(current["weather_code"], "mixed conditions")
        temp = round(current["temperature_2m"])
        return Response(
            f"It's {temp} degrees Celsius with {sky} in {place['name']}.",
            data={"temperature_c": temp, "city": place["name"]},
        )
    except Exception:  # network down, unknown city, API change: degrade gracefully
        url = _search(ctx, f"weather in {city}")
        return Response(
            f"I couldn't reach the weather service, so I opened a search for {city}.", url=url
        )


@skill("general_qa", "dictionary")
def web_answer(text: str, ctx: SkillContext) -> Response:
    query = normalize_text(text)
    url = _search(ctx, query)
    return Response("Here's what I found on the web.", url=url)


@skill("translation")
def translate(text: str, ctx: SkillContext) -> Response:
    phrase = ent.extract_after(text, ("translate", "how do you say", "how to say")) or text
    url = "https://translate.google.com/?op=translate&text=" + urllib.parse.quote(phrase)
    ctx.open_url(url)
    return Response("I've opened the translation for you.", url=url)


@skill("calculator")
def calculate(text: str, ctx: SkillContext) -> Response:
    try:
        result = ent.parse_arithmetic(text)
    except ZeroDivisionError:
        return Response("I can't divide by zero.")
    if result is None:
        return Response("Sorry, I couldn't find a calculation in that. Try '12 times 4'.")
    return Response(f"The answer is {ent.format_number(result)}.", data={"result": result})


@skill("unit_conversion")
def convert(text: str, ctx: SkillContext) -> Response:
    parsed = ent.parse_conversion(text)
    if parsed is None:
        return Response("Tell me the value and both units, for example '5 km to miles'.")
    value, src, result, dst = parsed
    return Response(
        f"{ent.format_number(value)} {src} is {ent.format_number(round(result, 3))} {dst}.",
        data={"result": result},
    )


@skill("news")
def news(text: str, ctx: SkillContext) -> Response:
    topic = ent.extract_after(text, ("news about", "news on", "headlines about"))
    if topic:
        url = _search(ctx, topic, "https://news.google.com/search?q=")
        return Response(f"Here are the latest headlines about {topic}.", url=url)
    url = "https://news.google.com/"
    ctx.open_url(url)
    return Response("Here are today's top headlines.", url=url)


@skill("traffic", "directions")
def directions(text: str, ctx: SkillContext) -> Response:
    destination = ent.extract_destination(text) or ent.extract_place(text)
    if not destination:
        return Response("Where would you like to go?")
    url = "https://www.google.com/maps/dir/?api=1&destination=" + urllib.parse.quote(destination)
    ctx.open_url(url)
    return Response(f"Here's the route and live traffic to {destination}.", url=url)


FACTS = [
    "Honey never spoils; edible honey has been found in ancient Egyptian tombs.",
    "Octopuses have three hearts and blue blood.",
    "Botanically, a tomato is a fruit.",
    "A day on Venus is longer than a year on Venus.",
    "Bananas are berries, but strawberries are not.",
]


@skill("facts")
def fact(text: str, ctx: SkillContext) -> Response:
    return Response(random.choice(FACTS))


# --- media & web ------------------------------------------------------------------------


def _media_query(text: str, *verbs: str) -> str | None:
    query = ent.extract_after(text, verbs)
    if query and query not in {"music", "some music", "a song", "something", "song"}:
        return query.removeprefix("some ").strip()
    return None


@skill("play_music")
def play_music(text: str, ctx: SkillContext) -> Response:
    query = _media_query(text, "play", "put on", "listen to")
    if not query:
        favourite = ctx.memory["preferences"].get("favorite_music")
        query = f"{favourite} music" if favourite else "popular music"
    url = _search(ctx, query, "https://www.youtube.com/results?search_query=")
    return Response(f"Playing {query} on YouTube.", url=url)


@skill("play_radio")
def play_radio(text: str, ctx: SkillContext) -> Response:
    station = _media_query(text, "play", "tune in to", "listen to") or "live radio"
    url = _search(ctx, station, "https://radio.garden/search?q=")
    return Response(f"Finding {station}.", url=url)


@skill("podcast_audiobook")
def podcast(text: str, ctx: SkillContext) -> Response:
    query = _media_query(text, "play", "listen to", "resume") or "popular podcasts"
    url = _search(ctx, query, "https://open.spotify.com/search/")
    return Response(f"Searching Spotify for {query}.", url=url)


@skill("movies")
def movies(text: str, ctx: SkillContext) -> Response:
    url = "https://www.imdb.com/chart/moviemeter/"
    ctx.open_url(url)
    return Response("Here are the most popular movies right now.", url=url)


@skill("open_website")
def open_website(text: str, ctx: SkillContext) -> Response:
    site = ent.extract_website(text)
    if site is None:
        return Response("Which website would you like me to open?")
    name, url = site
    ctx.open_url(url)
    return Response(f"Opening {name}.", url=url)


# --- lists & memory ---------------------------------------------------------------------


def _list_skill(text: str, ctx: SkillContext, key: str, label: str) -> Response:
    items = ctx.memory[key]
    if _has(text, "clear", "empty", "delete everything"):
        ctx.memory[key] = []
        return Response(f"Your {label} is now empty.")
    item = ent.extract_list_item(text)
    if item and not _has(text, "what", "read", "show"):
        added = ent.split_items(item)
        for entry in added:
            ctx.memory.append(key, entry)
        return Response(f"Added {_join(added)} to your {label}.")
    if not items:
        return Response(f"Your {label} is empty.")
    return Response(f"Your {label} has: " + ", ".join(items) + ".", data={"items": list(items)})


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


EXPORT_WORDS = ("document", "doc", "docx", "word", "file", "export", "print", "save", "write")


@skill("shopping_list")
def shopping_list(text: str, ctx: SkillContext) -> Response:
    if _has(text, *EXPORT_WORDS):
        return export_shopping_list(text, ctx)
    return _list_skill(text, ctx, "shopping_list", "shopping list")


def export_shopping_list(text: str, ctx: SkillContext) -> Response:
    """'save my shopping list as a word document' or
    'make a shopping list document with milk, eggs and bread'."""
    new_items = ent.extract_items_after_with(text)
    for entry in new_items:
        ctx.memory.append("shopping_list", entry)
    items = list(ctx.memory["shopping_list"])
    if not items:
        return Response("Your shopping list is empty. Tell me what to add first.")
    path = write_shopping_list(items, ctx.documents_dir, ctx.now(), ctx.memory["name"])
    ctx.open_file(path)
    count = f"{len(items)} item{'s' if len(items) != 1 else ''}"
    return Response(
        f"I saved your shopping list with {count} as a Word document and opened it.",
        data={"path": str(path), "items": items},
    )


@skill("todo_task")
def todo(text: str, ctx: SkillContext) -> Response:
    return _list_skill(text, ctx, "todos", "to-do list")


@skill("memory")
def recall(text: str, ctx: SkillContext) -> Response:
    m = ctx.memory
    facts = []
    if m["name"]:
        facts.append(f"your name is {m['name']}")
    for key, value in m["preferences"].items():
        facts.append(f"your {key.replace('_', ' ')} is {value}")
    if m["shopping_list"]:
        facts.append("your shopping list has " + ", ".join(m["shopping_list"]))
    if not facts:
        return Response("I don't know much about you yet. Try 'my name is ...'.")
    return Response("I remember that " + "; ".join(facts) + ".")


# --- smart home (simulated device registry) ------------------------------------------------

DEVICES = ("lights", "light", "fan", "tv", "heater", "air conditioner", "ac", "thermostat")


@skill("smart_home")
def smart_home(text: str, ctx: SkillContext) -> Response:
    norm = normalize_text(text)
    device = next((d for d in DEVICES if f" {d} " in f" {norm} "), None)
    if device is None:
        return Response("Which device should I control?")
    device = "lights" if device == "light" else device
    room_match = re.search(r"\bin\s+(?:the\s+)?([a-z]+(?:\s+room)?)\b", norm)
    room = room_match.group(1) if room_match else "home"
    state = "off" if _has(norm, "off", "stop", "disable") else "on"
    devices = dict(ctx.memory["devices"])
    devices[f"{room.lower()}:{device}"] = state
    ctx.memory["devices"] = devices
    where = f" in the {room}" if room != "home" else ""
    return Response(f"Turned {state} the {device}{where}.")


@skill("device_status")
def device_status(text: str, ctx: SkillContext) -> Response:
    devices = ctx.memory["devices"]
    if not devices:
        return Response("No smart devices have been used yet.")
    states = [f"{k.split(':')[1]} in {k.split(':')[0]} is {v}" for k, v in devices.items()]
    return Response("; ".join(states).capitalize() + ".")


# --- small talk ----------------------------------------------------------------------------

JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "Why did the computer get cold? It left its Windows open.",
    "I would tell you a UDP joke, but you might not get it.",
]
RIDDLES = [
    "What has keys but can't open locks? A piano.",
    "What gets wetter the more it dries? A towel.",
    "What has a head and a tail but no body? A coin.",
]


@skill("jokes_fun")
def joke(text: str, ctx: SkillContext) -> Response:
    return Response(random.choice(JOKES))


@skill("games_trivia")
def trivia(text: str, ctx: SkillContext) -> Response:
    return Response(random.choice(RIDDLES))


@skill("personality")
def personality(text: str, ctx: SkillContext) -> Response:
    norm = normalize_text(text)
    if "your name" in norm or "who are you" in norm:
        return Response("I'm an offline voice assistant built with Whisper and scikit-learn.")
    if "how are you" in norm:
        return Response("I'm running smoothly, thanks for asking.")
    return Response("I'm here to help with timers, lists, the weather and more.")
