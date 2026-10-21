import datetime as dt

import pytest

from voice_assistant import entities as ent


@pytest.mark.parametrize(
    "text, seconds",
    [
        ("set a timer for 10 minutes", 600),
        ("timer for five minutes", 300),
        ("1 hour 30 minutes", 5400),
        ("half an hour please", 1800),
        ("30 seconds", 30),
        ("start a timer", None),
    ],
)
def test_parse_duration(text, seconds):
    assert ent.parse_duration(text) == seconds


@pytest.mark.parametrize(
    "text, expected",
    [
        ("wake me up at 7 pm", dt.time(19, 0)),
        ("set an alarm for 7:30 am", dt.time(7, 30)),
        ("alarm at 6", dt.time(6, 0)),
        ("12 am", dt.time(0, 0)),
        ("at noon", dt.time(12, 0)),
        ("19:15", dt.time(19, 15)),
        ("remind me in 5 minutes", None),
        ("wake me up", None),
    ],
)
def test_parse_time(text, expected):
    assert ent.parse_time(text) == expected


@pytest.mark.parametrize(
    "text, result",
    [
        ("what is 5 plus 3", 8),
        ("12 times 4?", 48),
        ("100 divided by 8", 12.5),
        ("what's ten minus twelve", -2),
        ("hello there", None),
    ],
)
def test_parse_arithmetic(text, result):
    assert ent.parse_arithmetic(text) == result


def test_division_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        ent.parse_arithmetic("5 divided by 0")


def test_conversions():
    assert ent.parse_conversion("convert 5 km to miles")[2] == pytest.approx(3.10686, 1e-4)
    assert ent.parse_conversion("100 fahrenheit to celsius")[2] == pytest.approx(37.7778, 1e-4)
    assert ent.parse_conversion("5 km to kg") is None  # incompatible dimensions


def test_places_and_items():
    assert ent.extract_place("what's the weather in new york today") == "New York"
    assert ent.extract_destination("how do I get to the airport") == "Airport"
    assert ent.extract_list_item("Add milk and eggs to my shopping list") == "milk and eggs"
    assert ent.extract_list_item("I need to buy some apples") == "apples"


def test_website():
    assert ent.extract_website("open you tube") == ("Youtube", "https://www.youtube.com")
    assert ent.extract_website("go to example.org") == ("example.org", "https://example.org")
    assert ent.extract_website("open something") is None


def test_item_splitting():
    assert ent.split_items("milk, eggs and bread") == ["milk", "eggs", "bread"]
    assert ent.split_items("some apples") == ["apples"]
    assert ent.extract_items_after_with(
        "Make a shopping list with pasta, tomatoes and basil in a Word document please"
    ) == ["pasta", "tomatoes", "basil"]
    assert ent.extract_items_after_with("save my shopping list") == []
