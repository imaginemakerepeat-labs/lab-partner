# skills.py

import os
import shutil
import datetime
from web_search import search_web, format_results


def run_skill(text: str):
    """
    Returns (handled, response)
    handled=True means the skill responded and LLM should be skipped
    """

    if not text:
        return False, None

    t = text.lower().strip()

    # ----------------
    # DATE + TIME
    # ----------------
    if "date and time" in t or "time and date" in t:
        now = datetime.datetime.now()
        current_time = now.strftime("%I:%M %p").lstrip("0")
        current_date = now.strftime("%A, %B %d, %Y").replace(" 0", " ")
        return True, f"It is {current_time} on {current_date}."

    # ----------------
    # DATE
    # ----------------
    if (
        "what is the date" in t
        or "what's the date" in t
        or "current date" in t
        or "today's date" in t
        or "todays date" in t
        or "what day is it" in t
        or "what day is today" in t
        or t == "date"
    ):
        now = datetime.datetime.now()
        current_date = now.strftime("%A, %B %d, %Y").replace(" 0", " ")
        return True, f"Today is {current_date}."

    # ----------------
    # TIME
    # ----------------
    if (
        "what time is it" in t
        or "current time" in t
        or "tell me the time" in t
        or "local time" in t
        or "time is it" in t
        or t == "time"
    ):
        now = datetime.datetime.now().strftime("%I:%M %p").lstrip("0")
        return True, f"The time is {now}."

    # ----------------
    # DISK
    # ----------------
    if "disk" in t or "storage" in t:
        total, used, free = shutil.disk_usage("/")
        percent = int(used / total * 100)
        free_gb = free // (1024 ** 3)
        return True, f"Disk usage is {percent} percent, with about {free_gb} gigabytes free."

    # ----------------
    # HOSTNAME
    # ----------------
    if "hostname" in t or "host name" in t:
        return True, f"This system is {os.uname().nodename}."

    # ----------------
    # WEB SEARCH
    # ----------------
    prefixes = [
        "search the web for ",
        "look up ",
        "search online for ",
        "look online for ",
        "do a web search for ",
        "do a web search on ",
        "web search for ",
        "web search on ",
    ]

    matched_prefix = None
    for prefix in prefixes:
        if t.startswith(prefix):
            matched_prefix = prefix
            break

    if matched_prefix:
        query = t[len(matched_prefix):].strip(" .,!?")

        if not query:
            return True, "What would you like me to search for?"

        blocked_terms = ["porn", "sex", "torrent", "pirate"]
        for b in blocked_terms:
            if b in query:
                return True, "I cannot search for that."

        try:
            results = search_web(query, max_results=3)
            formatted = format_results(results)
            return True, f"I searched the web for '{query}'. Here are the top results:\n\n{formatted}"
        except Exception as e:
            return True, f"I tried to search the web but encountered an error: {e}"

    # ----------------
    # NOTHING MATCHED
    # ----------------
    return False, None