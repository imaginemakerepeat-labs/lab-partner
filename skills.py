# skills.py

import os
import shutil
import datetime


def run_skill(text: str):
    """
    Returns (handled, response)
    handled=True means the skill responded and LLM should be skipped
    """

    t = text.lower()

    # ----------------
    # TIME
    # ----------------
    if "time" in t:
        now = datetime.datetime.now().strftime("%I:%M %p")
        return True, f"The time is {now}."

    # ----------------
    # DISK
    # ----------------
    if "disk" in t or "storage" in t:
        total, used, free = shutil.disk_usage("/")
        percent = int(used / total * 100)
        return True, f"Disk usage is {percent} percent."

    # ----------------
    # HOSTNAME
    # ----------------
    if "hostname" in t or "host name" in t:
        return True, f"This system is {os.uname().nodename}."

    # ----------------
    # NOTHING MATCHED
    # ----------------
    return False, None
