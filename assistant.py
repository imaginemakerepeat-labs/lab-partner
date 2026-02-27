#!/usr/bin/env python3
"""
Lab Partner — Raspberry Pi Dual PTT + Interrupt
Red = OpenAI
Green = Ollama
Yellow (BTN_BASE2) = Interrupt
"""

import os
import json
import tempfile
import subprocess
import threading
from pathlib import Path
from datetime import datetime
import requests

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI
from evdev import InputDevice, ecodes


# ----------------------------
# DEFAULT CONFIG
# ----------------------------

DEFAULT_CONFIG = {
    "device_path": "/dev/input/by-id/usb-0079_USB_Gamepad-event-joystick",
    "sample_rate": 16000,

    "models": {
        "stt": "whisper-1",
        "chat": "gpt-4o"
    },

    "tts": {
        "engine": "espeak-ng",
        "voice": "en-us",
        "rate": 175
    },

    "logging": {
        "enabled": True,
        "path": "logs/conversation.jsonl",
        "max_turns_in_memory": 12
    },

    "context": {
        "local_turns": 6
    },

    "buttons": {
        "red_ptt": "BTN_THUMB",
        "green_ptt": "BTN_TOP",
        "interrupt": "BTN_BASE2"
    },

    "routes": {
        "red": {
            "backend": "openai",
            "chat_model": "gpt-4o"
        },
        "green": {
            "backend": "ollama",
            "ollama_model": "qwen2.5:1.5b-instruct"
        }
    }
}


# ----------------------------
# CONFIG
# ----------------------------

def _deep_merge(base, override):
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _ecode(name):
    if not hasattr(ecodes, name):
        raise ValueError(f"Unknown button '{name}'")
    return getattr(ecodes, name)


def load_config(path="config.json"):
    config = dict(DEFAULT_CONFIG)
    p = Path(path)

    if p.exists():
        with p.open("r") as f:
            config = _deep_merge(config, json.load(f))

    buttons = config["buttons"]
    config["red_key"] = _ecode(buttons["red_ptt"])
    config["green_key"] = _ecode(buttons["green_ptt"])
    config["interrupt_key"] = _ecode(buttons["interrupt"])

    return config


def append_jsonl(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


# ----------------------------
# INIT
# ----------------------------

load_dotenv()
client = OpenAI()

cfg = load_config()

DEVICE_PATH = cfg["device_path"]
SAMPLE_RATE = cfg["sample_rate"]
LOCAL_TURNS = cfg["context"]["local_turns"]

RED_KEY = cfg["red_key"]
GREEN_KEY = cfg["green_key"]
INTERRUPT_KEY = cfg["interrupt_key"]

ROUTES = cfg["routes"]

LOG_ENABLED = cfg["logging"]["enabled"]
LOG_PATH = cfg["logging"]["path"]
MAX_TURNS = cfg["logging"]["max_turns_in_memory"]

TTS_ENGINE = cfg["tts"]["engine"]
TTS_VOICE = cfg["tts"]["voice"]
TTS_RATE = cfg["tts"]["rate"]

messages = []

# interrupt control
cancel_turn = threading.Event()
tts_proc = None
tts_lock = threading.Lock()


# ----------------------------
# AUDIO
# ----------------------------

def record_audio(stop_event):
    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        while not stop_event.is_set():
            sd.sleep(50)

    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)

    return np.concatenate(chunks, axis=0)


def transcribe(audio):
    if audio.size == 0:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=cfg["models"]["stt"],
                file=f
            )
        return resp.text.strip()
    finally:
        os.unlink(tmp_path)


# ----------------------------
# CHAT
# ----------------------------

def _trim():
    global messages
    messages = messages[-MAX_TURNS:]


def chat_openai(text, model):
    messages.append({"role": "user", "content": text})
    _trim()

    resp = client.chat.completions.create(
        model=model,
        messages=messages
    )

    reply = resp.choices[0].message.content.strip()
    messages.append({"role": "assistant", "content": reply})
    _trim()
    return reply


def _local_prompt():
    lines = ["You are Lab Partner, a concise workshop assistant.\n"]

    for m in messages[-LOCAL_TURNS:]:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")

    lines.append("Assistant:")
    return "\n".join(lines)


def chat_ollama(text, model):
    messages.append({"role": "user", "content": text})
    _trim()

    prompt = _local_prompt()
    print("Thinking (local)...")

    r = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    r.raise_for_status()

    reply = r.json().get("response", "").strip()
    messages.append({"role": "assistant", "content": reply})
    _trim()
    return reply


def generate(text, route):
    backend = route["backend"]
    if backend == "openai":
        return chat_openai(text, route["chat_model"]), "openai"
    if backend == "ollama":
        return chat_ollama(text, route["ollama_model"]), "ollama"
    raise ValueError("Unknown backend")


# ----------------------------
# INTERRUPTIBLE SPEECH
# ----------------------------

def speak(text):
    global tts_proc
    if not text:
        return

    with tts_lock:
        if tts_proc and tts_proc.poll() is None:
            tts_proc.terminate()

        tts_proc = subprocess.Popen(
            [TTS_ENGINE, "-v", TTS_VOICE, "-s", str(TTS_RATE), text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


def interrupt():
    cancel_turn.set()
    global tts_proc

    with tts_lock:
        if tts_proc and tts_proc.poll() is None:
            tts_proc.terminate()

    print("(interrupted)")


# ----------------------------
# MAIN LOOP
# ----------------------------

print("==============================================")
print(" Lab Partner — Dual PTT + Interrupt")
print("==============================================")
print("Red = OpenAI | Green = Local | Yellow = Interrupt")
print("Ctrl+C to quit.\n")

device = InputDevice(DEVICE_PATH)

is_recording = False
stop_event = None
audio = None
active_route = None


try:
    for event in device.read_loop():

        if event.type != ecodes.EV_KEY:
            continue

        if event.code == INTERRUPT_KEY and event.value == 1:
            interrupt()
            continue

        if event.code in (RED_KEY, GREEN_KEY):

            if event.value == 1:  # press
                cancel_turn.clear()
                is_recording = True
                stop_event = threading.Event()

                active_route = ROUTES["red"] if event.code == RED_KEY else ROUTES["green"]

                def runner():
                    global audio
                    audio = record_audio(stop_event)

                threading.Thread(target=runner, daemon=True).start()

            elif event.value == 0 and is_recording:  # release
                is_recording = False
                stop_event.set()

                text = transcribe(audio)
                if not text:
                    continue

                print("\nYou:", text)

                reply, backend = generate(text, active_route)

                if cancel_turn.is_set():
                    print("(cancelled turn — ignoring reply)")
                    continue

                print(f"Assistant ({backend}): {reply}\n")

                speak(reply)

except KeyboardInterrupt:
    print("\nExiting...")