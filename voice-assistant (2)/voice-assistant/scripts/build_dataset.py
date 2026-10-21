"""Build data/alexa_data.csv: a varied, de-duplicated intent dataset.

Each intent has many hand-written templates with slots ({city}, {time}, {item}, ...).
Templates are expanded with slot values and optional politeness wrappers
("hey alexa", "can you ...", "... please"), then sampled without replacement so
every sentence is unique. Sentences from data/heldout_utterances.csv are excluded,
so the held-out evaluation measures phrasing the model has never seen.

Usage:  python scripts/build_dataset.py [--per-intent 200] [--seed 7]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SLOTS = {
    "time": ["7 am", "6:30 am", "7:45 am", "8 pm", "noon", "midnight", "5:15", "9 o'clock",
             "6 in the morning", "10:30 pm", "quarter past seven", "half past eight",
             "11 am", "4 pm", "seven thirty", "six"],
    "day": ["tomorrow", "today", "monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday", "the weekend", "next week", "tonight"],
    "duration": ["5 minutes", "ten minutes", "an hour", "30 seconds", "two hours",
                 "1 hour and 15 minutes", "90 seconds", "half an hour", "3 minutes",
                 "twenty minutes", "45 minutes", "a minute", "15 seconds", "25 minutes"],
    "city": ["london", "paris", "belgrade", "new york", "tokyo", "berlin", "madrid", "rome",
             "chicago", "sydney", "toronto", "vienna", "novi sad", "amsterdam", "dubai",
             "los angeles", "moscow", "istanbul", "zagreb", "athens", "seattle", "miami"],
    "item": ["milk", "eggs", "bread", "butter", "apples", "bananas", "coffee", "rice",
             "chicken", "tomatoes", "cheese", "yogurt", "pasta", "olive oil", "toilet paper",
             "dish soap", "orange juice", "onions", "potatoes", "flour", "sugar", "tea",
             "batteries", "shampoo", "cereal", "garlic", "carrots", "water bottles"],
    "task": ["call the bank", "finish the report", "pay the electricity bill",
             "clean the kitchen", "book a dentist appointment", "send the invoice",
             "study for the exam", "walk the dog", "renew my passport", "reply to emails",
             "pick up the kids", "fix the bike", "water the garden", "buy a birthday gift",
             "submit the assignment", "do the laundry", "call grandma", "back up my laptop"],
    "genre": ["jazz", "rock", "metal", "classical", "hip hop", "lo-fi", "pop", "country",
              "blues", "reggae", "techno", "indie", "acoustic", "eighties"],
    "artist": ["the beatles", "metallica", "taylor swift", "queen", "daft punk", "adele",
               "eminem", "coldplay", "nirvana", "beyonce", "pink floyd", "drake",
               "iron maiden", "billie eilish", "ac dc", "bob marley"],
    "site": ["youtube", "google", "github", "gmail", "wikipedia", "linkedin", "netflix",
             "reddit", "stack overflow", "spotify", "amazon", "facebook", "twitter",
             "instagram", "bbc.com", "kaggle.com"],
    "device": ["lights", "light", "fan", "tv", "heater", "air conditioner", "lamp",
               "thermostat", "speaker", "coffee machine", "vacuum", "plug"],
    "room": ["kitchen", "living room", "bedroom", "bathroom", "office", "hallway", "garage",
             "kids room", "dining room"],
    "word": ["serendipity", "ephemeral", "ubiquitous", "melancholy", "algorithm",
             "paradigm", "esoteric", "cogent", "benevolent", "quintessential", "ambiguous",
             "eloquent", "meticulous", "pragmatic", "resilient", "obfuscate", "nostalgia",
             "ubiquity", "verbose", "candid", "frugal", "lucid", "tenacious", "zealous"],
    "phrase": ["hello", "thank you", "good morning", "where is the bathroom",
               "how much does this cost", "i love you", "see you later", "good night",
               "nice to meet you", "excuse me", "i am hungry", "happy birthday", "cheers",
               "how are you", "goodbye", "what time is it"],
    "lang": ["spanish", "french", "german", "italian", "japanese", "serbian", "russian",
             "portuguese", "chinese", "korean", "greek", "dutch", "turkish", "arabic"],
    "dest": ["the airport", "the train station", "work", "the office", "downtown",
             "the nearest gas station", "the hospital", "the mall", "home", "the beach",
             "the university", "the city center", "the stadium", "the nearest pharmacy",
             "the supermarket", "the bus station", "the museum", "my parents house"],
    "topic": ["sports", "technology", "politics", "business", "science", "football",
              "the stock market", "health", "entertainment", "the economy", "space",
              "climate", "basketball", "artificial intelligence", "europe"],
    "show": ["true crime", "history", "comedy", "self improvement", "science", "business",
             "tech", "philosophy", "football", "harry potter", "the hobbit", "dune"],
    "station": ["bbc radio one", "jazz fm", "npr", "classic rock radio", "kiss fm",
                "radio belgrade", "capital fm", "the local news station", "a country station",
                "smooth radio"],
    "question": ["who is the president of france", "how tall is mount everest",
                 "what is the capital of australia", "who invented the light bulb",
                 "how far away is the moon", "when did world war two end",
                 "how many continents are there", "what is the speed of light",
                 "who painted the mona lisa", "how old is the universe",
                 "what is the largest ocean", "who discovered penicillin",
                 "how many bones are in the human body", "what is the population of china",
                 "who wrote harry potter", "why is the sky blue",
                 "how does photosynthesis work", "what is the longest river in the world",
                 "who was nikola tesla", "how deep is the ocean",
                 "what language do they speak in brazil", "how many players on a football team"],
}  # fmt: skip

_NUMS = list(range(2, 100))
_UNIT_PAIRS = [("km", "miles"), ("miles", "kilometers"), ("kilograms", "pounds"),
               ("pounds", "kg"), ("celsius", "fahrenheit"), ("fahrenheit", "celsius"),
               ("meters", "feet"), ("feet", "meters"), ("liters", "gallons"),
               ("inches", "centimeters"), ("ounces", "grams"), ("grams", "ounces"),
               ("gallons", "liters"), ("centimeters", "inches")]  # fmt: skip


def _calc_slots(rng: random.Random) -> list[dict]:
    ops = ["plus", "minus", "times", "divided by", "multiplied by", "+", "x"]
    return [{"a": rng.choice(_NUMS), "op": rng.choice(ops), "b": rng.choice(_NUMS)}
            for _ in range(120)]  # fmt: skip


def _unit_slots(rng: random.Random) -> list[dict]:
    return [{"n": rng.choice([1, 2, 5, 10, 12, 20, 25, 30, 50, 100, 3.5]), "u1": a, "u2": b}
            for a, b in _UNIT_PAIRS for _ in range(6)]  # fmt: skip


TEMPLATES: dict[str, list[str]] = {
    "alarm": [
        "set an alarm for {time}", "wake me up at {time}", "alarm at {time} {day}",
        "i need to be up at {time}", "set my alarm to {time}", "wake me at {time} {day}",
        "create an alarm for {time}", "get me up at {time}", "set a wake up alarm for {time}",
        "cancel my alarm", "turn off the alarm", "delete all my alarms",
        "change my alarm to {time}", "snooze the alarm", "what time is my alarm set for",
        "i have to wake up at {time} {day}", "make an alarm for {time} on {day}",
        "set an alarm", "wake me up early {day}", "alarm please",
    ],
    "calculator": [
        "what is {a} {op} {b}", "calculate {a} {op} {b}", "what's {a} {op} {b}",
        "how much is {a} {op} {b}", "{a} {op} {b}", "can you work out {a} {op} {b}",
        "solve {a} {op} {b}", "add {a} and {b}", "subtract {b} from {a}",
        "multiply {a} by {b}", "divide {a} by {b}", "what is {a} percent of {b}",
        "what is the square root of {a}", "compute {a} {op} {b}",
    ],
    "calendar": [
        "what's on my calendar {day}", "do i have any meetings {day}",
        "show me my schedule for {day}", "what does my day look like",
        "am i free {day}", "any appointments {day}", "what's my agenda for {day}",
        "check my calendar", "add a meeting {day} at {time}",
        "schedule a call with the team {day}", "when is my next meeting",
        "what events do i have this week", "is anything planned for {day}",
        "put dinner with friends on my calendar for {day}", "cancel my meeting {day}",
        "how busy am i {day}",
    ],
    "date_time": [
        "what time is it", "what's the time", "tell me the time", "what's the date today",
        "what day is it", "what is today's date", "current time", "what time is it in {city}",
        "what year is it", "what month is it", "is it {day} today", "do you know what time it is",
        "how many days until {day}", "what day of the week is it", "what's the date",
        "time please", "what's the time in {city} right now",
    ],
    "device_status": [
        "are the {device} on", "is the {device} in the {room} on", "did i leave the {device} on",
        "status of the {device}", "which devices are on", "what's the status of my devices",
        "is the {device} still running", "check the {device} in the {room}",
        "are any lights on in the house", "what's turned on right now", "is the {device} off",
        "show me my smart home status", "what is the thermostat set to",
        "is the {room} {device} switched on",
    ],
    "dictionary": [
        "define {word}", "what does {word} mean", "meaning of {word}",
        "what is the definition of {word}", "how do you spell {word}", "spell {word}",
        "give me a synonym for {word}", "what's another word for {word}",
        "what's the opposite of {word}", "look up {word} in the dictionary",
        "use {word} in a sentence", "is {word} a real word",
        "what's the meaning of the word {word}",
    ],
    "directions": [
        "directions to {dest}", "how do i get to {dest}", "navigate to {dest}",
        "take me to {dest}", "route to {dest}", "show me the way to {dest}",
        "how far is {dest}", "find the fastest route to {dest}", "how long to walk to {dest}",
        "get directions to {dest}", "what's the best way to get to {dest}",
        "start navigation to {dest}", "where is {dest}", "drive me to {dest}",
    ],
    "facts": [
        "tell me a fact", "tell me a random fact", "tell me something interesting",
        "give me a fun fact", "share a fun fact", "teach me something new",
        "tell me a fact about {topic}", "give me an interesting fact about animals",
        "do you know any cool facts", "fact of the day", "surprise me with a fact",
        "tell me something i probably don't know", "random fact about space",
        "tell me a fun fact about history",
    ],
    "games_trivia": [
        "let's play a game", "ask me a riddle", "give me a trivia question",
        "play trivia", "quiz me", "start a quiz about {topic}", "let's play twenty questions",
        "tell me a riddle", "can we play a word game", "play rock paper scissors",
        "test my knowledge", "give me a brain teaser", "i want to play a game",
        "quiz me on {topic}", "trivia time",
    ],
    "general_qa": ["{question}", "tell me {question}", "do you know {question}",
                   "i want to know {question}", "quick question {question}"],
    "jokes_fun": [
        "tell me a joke", "make me laugh", "say something funny", "do you know any jokes",
        "tell me a funny joke", "tell me a dad joke", "joke please", "tell me a programming joke",
        "give me a pun", "i need a laugh", "know any good jokes", "tell me another joke",
        "say a knock knock joke", "cheer me up", "tell me a joke about {topic}",
    ],
    "memory": [
        "what do you remember about me", "what do you know about me", "do you remember my name",
        "what's my name", "what did i tell you about myself", "what have you stored about me",
        "do you know who i am", "what's my favorite music", "what do you have saved about me",
        "remind me what you know about me", "what information do you have on me",
        "do you remember what music i like", "tell me what you remember",
    ],
    "movies": [
        "recommend a movie", "what movies are playing near me", "what are the top movies",
        "suggest a {genre_movie} movie", "what should i watch tonight", "show me popular films",
        "any good movies out", "what's new in cinemas", "find me a {genre_movie} film",
        "who stars in the latest batman movie", "best movies of all time",
        "recommend something to watch", "movie showtimes for {day}", "what's trending on netflix",
    ],
    "news": [
        "what's the news", "latest headlines", "news today", "tell me the news",
        "what's happening in the world", "give me the {topic} news", "news about {topic}",
        "any breaking news", "top stories today", "read me the headlines",
        "what's new in {topic}", "latest news from {city}", "morning news briefing",
        "what are today's top stories",
    ],
    "open_website": [
        "open {site}", "go to {site}", "launch {site}", "open the {site} website",
        "take me to {site}", "bring up {site}", "load {site}", "visit {site}",
        "can you open {site} for me", "show me {site}", "navigate to {site} website",
    ],
    "personality": [
        "how are you", "what is your name", "who made you", "are you a robot",
        "do you have feelings", "who are you", "what can you do", "are you smart",
        "do you like me", "how old are you", "what's your favorite color", "are you real",
        "where do you live", "do you sleep", "what are you", "are you happy",
        "thank you", "you're awesome", "good job",
    ],
    "play_music": [
        "play some {genre}", "play {genre} music", "play songs by {artist}", "play {artist}",
        "put on some {genre}", "i want to listen to {artist}", "play music",
        "play my playlist", "play something relaxing", "play a song", "shuffle my music",
        "play the latest album by {artist}", "next song", "play some {genre} for studying",
        "play my favorite songs", "turn on some music", "play upbeat music",
    ],
    "play_radio": [
        "play the radio", "play {station}", "tune in to {station}", "turn on the radio",
        "listen to {station}", "play {genre} radio", "start the radio", "put on {station}",
        "play a local radio station", "switch to {station}", "play the news radio",
    ],
    "podcast_audiobook": [
        "play a podcast", "play a {show} podcast", "resume my audiobook",
        "continue my audiobook", "play the latest episode", "next episode",
        "play the next chapter", "find a podcast about {show}", "play the {show} audiobook",
        "read me {show}", "play my podcasts", "resume the podcast",
        "what podcasts do you recommend",
    ],
    "reminder": [
        "remind me to {task}", "remind me to {task} at {time}", "remind me {day} to {task}",
        "set a reminder to {task}", "create a reminder for {time}",
        "don't let me forget to {task}", "add a reminder to {task}",
        "reminder to {task} {day}", "what are my reminders", "show my reminders",
        "delete my reminder", "remind me in {duration} to {task}",
        "can you remind me to {task}",
    ],
    "routine": [
        "good morning", "good night", "start my morning routine", "run my bedtime routine",
        "i'm home", "i'm leaving", "i'm going to bed", "start my evening routine",
        "good evening", "i'm off to work", "run my workout routine", "start my day",
        "time to sleep", "i'm back",
    ],
    "set_timer": [
        "set a timer for {duration}", "timer for {duration}", "start a {duration} timer",
        "countdown {duration}", "set a {duration} timer", "start a timer",
        "how much time is left on my timer", "cancel the timer", "stop the timer",
        "set a cooking timer for {duration}", "time {duration}",
        "start counting down from {duration}", "pause the timer",
    ],
    "shopping_list": [
        "add {item} to my shopping list", "put {item} on the shopping list",
        "add {item} to the list", "i need to buy {item}", "we need {item}",
        "add {item} and {item2} to my shopping list", "what's on my shopping list",
        "read my shopping list", "remove {item} from my shopping list",
        "clear my shopping list", "put {item} on my grocery list", "do i need to buy {item}",
        "show me the shopping list",
        "save my shopping list as a word document", "export my shopping list to a doc file",
        "make a shopping list document", "create a word doc with my shopping list",
        "make a shopping list with {item}, {item2} and {item3} in a document",
        "write my shopping list to a file", "put my shopping list in a word document",
        "print my shopping list", "create a shopping list document with {item} and {item2}",
        "save the grocery list to a docx", "generate a doc of my groceries",
    ],
    "smart_home": [
        "turn on the {device}", "turn off the {device}", "switch on the {device} in the {room}",
        "turn the {room} {device} off", "dim the lights in the {room}",
        "set the thermostat to {temp} degrees", "make it warmer", "make it cooler in here",
        "turn the {device} on in the {room}", "switch off all the lights",
        "lights on in the {room}", "brighten the {room} lights", "start the {device}",
        "turn down the heat", "lock the front door",
    ],
    "todo_task": [
        "add {task} to my to do list", "add a task to {task}", "what's on my to do list",
        "show my tasks", "mark {task} as done", "new task {task}", "i need to {task}",
        "what do i have to do {day}", "remove {task} from my tasks", "create a to do",
        "what tasks are left", "put {task} on my list of tasks", "clear my to do list",
    ],
    "traffic": [
        "how's the traffic", "how is the traffic to {dest}", "is there traffic on the way to work",
        "traffic report", "how long is my commute", "is the highway busy",
        "any traffic jams near me", "how bad is traffic in {city}", "should i leave now for work",
        "is there an accident on my route", "traffic update",
        "how long will it take to drive to {dest}",
    ],
    "translation": [
        "translate {phrase} to {lang}", "how do you say {phrase} in {lang}",
        "what is {phrase} in {lang}", "say {phrase} in {lang}", "translate {phrase} into {lang}",
        "how would i say {phrase} in {lang}", "what's the {lang} for {phrase}",
        "{phrase} in {lang}", "can you translate this to {lang}",
    ],
    "unit_conversion": [
        "convert {n} {u1} to {u2}", "how many {u2} is {n} {u1}", "{n} {u1} to {u2}",
        "what is {n} {u1} in {u2}", "how many {u2} in {n} {u1}", "change {n} {u1} into {u2}",
        "{n} {u1} equals how many {u2}",
    ],
    "weather": [
        "what's the weather", "what's the weather like in {city}", "weather in {city}",
        "will it rain {day}", "is it going to snow {day}", "how cold is it outside",
        "weather forecast for {day}", "what's the temperature in {city}", "do i need a jacket",
        "is it sunny in {city}", "what's the forecast for {city} {day}", "how hot will it be {day}",
        "will i need an umbrella {day}", "is it windy outside", "weather report",
    ],
}  # fmt: skip

EXTRA_SLOTS = {
    "item2": SLOTS["item"],
    "item3": SLOTS["item"],
    "genre_movie": ["comedy", "horror", "action", "romantic", "sci fi", "animated", "thriller",
                    "documentary", "family"],
    "temp": ["18", "20", "21", "22", "23", "24", "25"],
}  # fmt: skip

PREFIXES = ["", "", "", "", "hey alexa ", "alexa ", "please ", "can you ", "could you ",
            "hey ", "um ", "okay "]  # fmt: skip
SUFFIXES = ["", "", "", "", " please", " now", " for me", " thanks"]

_SLOT_RE = re.compile(r"\{(\w+)\}")


def expand(template: str, rng: random.Random, row_slots: list[dict] | None) -> str:
    if row_slots is not None:
        return template.format(**rng.choice(row_slots))
    values = {}
    for name in _SLOT_RE.findall(template):
        pool = SLOTS.get(name) or EXTRA_SLOTS[name]
        choice = rng.choice(pool)
        while name in {"item2", "item3"} and choice in (values.get("item"), values.get("item2")):
            choice = rng.choice(pool)
        values[name] = choice
    return template.format(**values)


def wrap(text: str, rng: random.Random) -> str:
    prefix, suffix = rng.choice(PREFIXES), rng.choice(SUFFIXES)
    # Questions read oddly with "can you"; keep only neutral prefixes for them.
    if prefix in {"can you ", "could you ", "please "} and text.split()[0] in {
        "what",
        "what's",
        "how",
        "who",
        "is",
        "are",
        "do",
        "did",
        "will",
        "when",
        "where",
        "which",
        "am",
        "any",
        "why",
        "should",
        "how's",
    }:
        prefix = ""
    return f"{prefix}{text}{suffix}".strip()


def build(per_intent: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    heldout_path = ROOT / "data" / "heldout_utterances.csv"
    banned = set()
    if heldout_path.exists():
        with heldout_path.open() as fh:
            banned = {_norm(r["prompt"]) for r in csv.DictReader(fh)}

    rows = []
    for intent, templates in TEMPLATES.items():
        row_slots = (
            _calc_slots(rng) if intent == "calculator"
            else _unit_slots(rng) if intent == "unit_conversion"
            else None
        )  # fmt: skip
        target = per_intent // 2 if intent == "memory" else per_intent
        seen: set[str] = set()
        # Cycle through templates so every phrasing is represented, not just the slot-heavy ones.
        for attempt, template in enumerate(itertools.cycle(templates)):
            if len(seen) >= target or attempt > target * 60:
                break
            text = wrap(expand(template, rng, row_slots), rng)
            key = _norm(text)
            if key in seen or key in banned:
                continue
            seen.add(key)
            rows.append((text, intent))
    rng.shuffle(rows)
    return rows


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9:'\s]", " ", text.lower())).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-intent", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "alexa_data.csv")
    args = parser.parse_args()

    rows = build(args.per_intent, args.seed)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["prompt", "intent"])
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for _, intent in rows:
        counts[intent] = counts.get(intent, 0) + 1
    print(f"Wrote {len(rows)} unique rows, {len(counts)} intents to {args.out}")
    print("Smallest classes:", sorted(counts.items(), key=lambda kv: kv[1])[:4])


if __name__ == "__main__":
    main()
