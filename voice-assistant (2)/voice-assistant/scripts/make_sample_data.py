"""Generate data/sample_intents.csv: a small synthetic dataset in the same format
as the original alexa_data.csv (templated prompts with the row index appended).

It exists so the project runs end to end, in CI and for anyone cloning the repo,
without the private dataset. Real results should be produced on the real data.
"""

import csv
import random
from pathlib import Path

TEMPLATES = {
    "alarm": ["wake me up", "set an alarm", "morning alarm", "set alarm for 6 am",
              "wake me at 7", "alarm for tomorrow morning", "set an alarm for 8 30"],
    "calculator": ["what is 5 plus 3", "calculate 12 times 4", "what's 100 divided by 5",
                   "add 7 and 9", "what is 20 minus 6", "multiply 3 by 8"],
    "calendar": ["what's on my calendar", "do i have meetings today", "show my schedule",
                 "any events tomorrow", "what is my agenda", "check my calendar"],
    "date_time": ["what time is it", "what's the date", "what day is it today",
                  "tell me the time", "current time please", "what is today's date"],
    "device_status": ["are the lights on", "is the tv on", "device status",
                      "which devices are on", "is the heater running", "status of my devices"],
    "dictionary": ["define serendipity", "what does ephemeral mean", "meaning of ubiquitous",
                   "definition of algorithm", "spell necessary", "synonym for happy"],
    "directions": ["directions to the airport", "how do i get to the station",
                   "navigate to downtown", "route to the hospital", "take me to the mall",
                   "directions home"],
    "facts": ["tell me a fact", "random fact", "tell me something interesting",
              "give me a fun fact", "did you know", "share a fact"],
    "games_trivia": ["let's play a game", "ask me a riddle", "trivia question",
                     "play trivia", "give me a quiz", "start a game"],
    "general_qa": ["who is the president of france", "how tall is mount everest",
                   "what is the capital of japan", "who invented the telephone",
                   "how far is the moon", "when was einstein born"],
    "jokes_fun": ["tell me a joke", "make me laugh", "say something funny",
                  "know any jokes", "tell a funny joke", "joke please"],
    "memory": ["what do you remember about me", "what do you know about me",
               "do you remember my name", "what did i tell you"],
    "movies": ["recommend a movie", "what movies are playing", "top movies",
               "movie suggestions", "what should i watch", "popular films"],
    "news": ["what's the news", "latest headlines", "news today", "tell me the news",
             "sports news", "top stories"],
    "open_website": ["open youtube", "open google", "go to github", "launch wikipedia",
                     "open reddit", "open my email"],
    "personality": ["how are you", "what is your name", "who made you", "are you a robot",
                    "do you have feelings", "who are you"],
    "play_music": ["play music", "play a song", "play some jazz", "play rock music",
                   "play my playlist", "play the beatles"],
    "play_radio": ["play the radio", "tune in to bbc radio", "play radio station",
                   "listen to fm radio", "play jazz radio", "start the radio"],
    "podcast_audiobook": ["play a podcast", "resume my audiobook", "play the latest episode",
                          "next podcast episode", "read my audiobook", "play a true crime podcast"],
    "reminder": ["remind me to call mom", "set a reminder", "remind me to buy milk",
                 "reminder for the meeting", "remind me at 5 pm to stretch", "add a reminder"],
    "routine": ["good morning", "good night", "start my morning routine",
                "run my evening routine", "i'm home", "i'm leaving"],
    "set_timer": ["set a timer for 10 minutes", "timer for 5 minutes", "start a timer",
                  "countdown 30 seconds", "set a 1 hour timer", "timer for 20 minutes"],
    "shopping_list": ["add milk to my shopping list", "what's on my shopping list",
                      "put eggs on the shopping list", "add bread to the list",
                      "clear my shopping list", "shopping list"],
    "smart_home": ["turn on the lights", "turn off the tv", "switch on the fan",
                   "dim the lights", "turn off the heater", "lights on in the kitchen"],
    "todo_task": ["add a task", "what's on my to do list", "add homework to my to do list",
                  "show my tasks", "mark task as done", "new to do"],
    "traffic": ["how's the traffic", "traffic to work", "is there traffic",
                "traffic report", "how long is my commute", "traffic on the highway"],
    "translation": ["translate hello to spanish", "how do you say thank you in french",
                    "translate good morning to german", "what is cat in italian",
                    "say goodbye in japanese", "translate this"],
    "unit_conversion": ["convert 5 km to miles", "how many feet in a meter",
                        "convert 100 fahrenheit to celsius", "10 pounds to kg",
                        "how many grams in an ounce", "convert 2 liters to gallons"],
    "weather": ["what's the weather", "weather today", "will it rain tomorrow",
                "weather in london", "is it cold outside", "temperature in paris"],
}  # fmt: skip


def main(rows_per_intent: int = 40, seed: int = 0) -> None:
    rng = random.Random(seed)
    rows = []
    for intent, templates in TEMPLATES.items():
        n = rows_per_intent // 4 if intent == "memory" else rows_per_intent
        rows += [(rng.choice(templates), intent) for _ in range(n)]
    rng.shuffle(rows)
    out = Path(__file__).resolve().parents[1] / "data" / "sample_intents.csv"
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["prompt", "intent"])
        for i, (prompt, intent) in enumerate(rows):
            writer.writerow([f"{prompt} {i}", intent])  # mimic the original row-id suffix
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
