#!/usr/bin/env python3
"""
Lab Partner — Dual PTT + Interrupt + Skills
"""

import threading
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from evdev import InputDevice, ecodes

from config import load_config, append_jsonl, read_text_file

from backends.openai_backend import chat_openai as backend_openai
from backends.ollama_backend import chat_ollama as backend_ollama

from audio import record_audio, transcribe
from tts import TTSController
from mouth import MouthController
from hud_controller import HudController
from memory import ChatMemory
from skills import run_skill


# ----------------------------
# INIT
# ----------------------------

load_dotenv()
client = OpenAI()

cfg = load_config()

DEVICE_PATH = cfg["device_path"]
SAMPLE_RATE = cfg["sample_rate"]

RED_KEY = cfg["red_key"]
GREEN_KEY = cfg["green_key"]
INTERRUPT_KEY = cfg["interrupt_key"]

ROUTES = cfg["routes"]

LOG_ENABLED = cfg["logging"]["enabled"]
LOG_PATH = cfg["logging"]["path"]
MAX_TURNS = cfg["logging"]["max_turns_in_memory"]

LOCAL_TURNS = cfg["context"]["local_turns"]

TTS_ENGINE = cfg["tts"]["engine"]
TTS_VOICE = cfg["tts"]["voice"]
TTS_RATE = cfg["tts"]["rate"]

OLLAMA_URL = cfg["ollama"]["url"]

# Maynard mouth
MAYNARD_ENABLED = cfg.get("maynard", {}).get("enabled", False)
MAYNARD_IP = cfg.get("maynard", {}).get("ip", "127.0.0.1")
MAYNARD_PORT = int(cfg.get("maynard", {}).get("port", 9000))

cancel_turn = threading.Event()

record_thread: Optional[threading.Thread] = None
audio_buffer = None

memory = ChatMemory(MAX_TURNS)


# ----------------------------
# HUD / MOUTH / TTS
# ----------------------------

hud = HudController()

mouth = MouthController(
    enabled=MAYNARD_ENABLED,
    ip=MAYNARD_IP,
    port=MAYNARD_PORT,
    interval=0.12,
)

tts = TTSController(
    engine=TTS_ENGINE,
    voice=TTS_VOICE,
    rate=TTS_RATE,
    on_speaking=lambda: (
        mouth.start_ticker(),
        hud.set("speaking"),
    ),
    on_idle=lambda: (
        mouth.stop_ticker(),
        hud.set("idle"),
    ),
    on_interrupt=lambda: (
        mouth.stop_ticker(),
        mouth.send("close", "interrupt"),
        mouth.send("clear", "interrupt"),
        hud.set("interrupted"),
    ),
)


# ----------------------------
# PERSONA / KNOWLEDGE
# ----------------------------

persona = read_text_file(cfg["prompt_files"]["persona"])
knowledge = read_text_file(cfg["prompt_files"]["knowledge_base"])

system_prompt = "\n".join([p for p in [persona, knowledge] if p]).strip()

if system_prompt:
    memory.set_system(system_prompt)


# ----------------------------
# CHAT
# ----------------------------

def generate(text: str, route: dict):
    backend = route["backend"]
    msgs = memory.list()

    if backend == "openai":
        reply = backend_openai(client, msgs, text, route["chat_model"])
        memory.trim()
        return reply, "openai"

    if backend == "ollama":
        reply = backend_ollama(
            msgs,
            text,
            route["ollama_model"],
            OLLAMA_URL,
            local_turns=LOCAL_TURNS,
        )
        memory.trim()
        return reply, "ollama"

    raise ValueError(f"Unknown backend: {backend}")


# ----------------------------
# INTERRUPT
# ----------------------------

def interrupt():
    cancel_turn.set()
    tts.interrupt()
    print("interrupted", flush=True)


# ----------------------------
# MAIN LOOP
# ----------------------------

print("==============================================")
print(" Lab Partner — Dual PTT + Interrupt + Skills")
print("==============================================")
print("Red = OpenAI | Green = Local | Yellow = Interrupt")
print("Ctrl+C to quit.\n")

hud.set("idle")

device = InputDevice(DEVICE_PATH)

is_recording = False
stop_event: Optional[threading.Event] = None
active_route = None

try:
    for event in device.read_loop():

        if event.type != ecodes.EV_KEY:
            continue

        # ----------------
        # INTERRUPT
        # ----------------
        if event.code == INTERRUPT_KEY and event.value == 1:
            interrupt()
            continue

        # ----------------
        # PTT KEYS
        # ----------------
        if event.code in (RED_KEY, GREEN_KEY):

            # PRESS
            if event.value == 1:

                cancel_turn.clear()

                is_recording = True
                stop_event = threading.Event()

                is_red = (event.code == RED_KEY)
                active_route = ROUTES["red"] if is_red else ROUTES["green"]

                label = "openai" if is_red else "local"

                print(f"recording ({label})...", flush=True)
                hud.set("recording", label)

                def runner():
                    global audio_buffer
                    audio_buffer = record_audio(stop_event, SAMPLE_RATE)

                record_thread = threading.Thread(target=runner, daemon=True)
                record_thread.start()

            # RELEASE
            elif event.value == 0 and is_recording:

                is_recording = False

                if stop_event:
                    stop_event.set()

                if record_thread:
                    record_thread.join(timeout=2.0)

                text = transcribe(
                    client,
                    audio_buffer,
                    SAMPLE_RATE,
                    cfg["models"]["stt"]
                )

                if not text:
                    hud.set("idle")
                    continue

                print("\nYou:", text, flush=True)

                if LOG_ENABLED:
                    append_jsonl(
                        LOG_PATH,
                        {"ts": time.time(), "role": "user", "text": text},
                    )

                # ----------------
                # SKILL ROUTER
                # ----------------

                handled, skill_reply = run_skill(text)

                if handled:

                    reply = skill_reply
                    backend = "skill"

                else:

                    backend_label = "openai" if active_route.get("backend") == "openai" else "local"
                    hud.set("thinking", backend_label)

                    reply, backend = generate(text, active_route)

                # ----------------
                # INTERRUPT CHECK
                # ----------------

                if cancel_turn.is_set():
                    print("(cancelled turn — ignoring reply)", flush=True)
                    hud.set("idle")
                    continue

                print(f"Assistant ({backend}): {reply}\n", flush=True)

                if LOG_ENABLED:
                    append_jsonl(
                        LOG_PATH,
                        {
                            "ts": time.time(),
                            "role": "assistant",
                            "backend": backend,
                            "text": reply
                        },
                    )

                tts.speak(reply)


except KeyboardInterrupt:

    print("\nExiting...", flush=True)

    try:
        hud.shutdown()
    except Exception:
        pass

    hud.set("idle")