#!/usr/bin/env python3
"""
Lab Partner — Dual PTT + Interrupt + Pygame HUD + Persona/KB (single-file, spawn-safe)

Red   = OpenAI (PTT hold)
Green = Ollama (PTT hold)
Yellow/Interrupt = BTN_PINKIE (press to interrupt speech + cancel current turn)

Files in repo root (same folder as assistant.py):
- persona.txt
- knowledge_base.txt

HUD:
- Optional pygame window in a separate process (spawn-safe)
- Updates on recording/thinking/speaking/interrupt/idle
- If pygame fails to start, assistant still runs normally
"""

import os
import json
import time
import tempfile
import subprocess
import threading
from pathlib import Path
import multiprocessing as mp

import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError
from evdev import InputDevice, ecodes


# ----------------------------
# PERSONA / KNOWLEDGE BASE
# ----------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def read_text_file(name: str) -> str:
    p = SCRIPT_DIR / name
    try:
        return p.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        print(f"Warning: {name} not found in {SCRIPT_DIR}", flush=True)
        return ""


# ----------------------------
# DEFAULT CONFIG
# ----------------------------

DEFAULT_CONFIG = {
    "device_path": "/dev/input/by-id/usb-0079_USB_Gamepad-event-joystick",
    "sample_rate": 16000,
    "models": {
        "stt": "whisper-1",
        "chat": "gpt-4o",
    },
    "tts": {
        "engine": "espeak-ng",
        "voice": "en-us",
        "rate": 175,
    },
    "logging": {
        "enabled": True,
        "path": "logs/conversation.jsonl",
        "max_turns_in_memory": 12,  # messages kept (not "turns")
    },
    "context": {
        "local_turns": 6,
    },
    "buttons": {
        "red_ptt": "BTN_THUMB",
        "green_ptt": "BTN_TOP",
        "interrupt": "BTN_PINKIE",
    },
    "routes": {
        "red": {"backend": "openai", "chat_model": "gpt-4o"},
        "green": {"backend": "ollama", "ollama_model": "qwen2.5:1.5b-instruct"},
    },
    "hud": {"enabled": True},
}


def _deep_merge(base, override):
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _ecode(name: str) -> int:
    if not hasattr(ecodes, name):
        raise ValueError(f"Unknown button '{name}' (not in evdev.ecodes)")
    return getattr(ecodes, name)


def load_config(path="config.json"):
    config = dict(DEFAULT_CONFIG)
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
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
# HUD (pygame in separate process)
# ----------------------------

HUD_STATE_IDLE = "IDLE"
HUD_STATE_RECORDING = "RECORDING"
HUD_STATE_THINKING = "THINKING"
HUD_STATE_SPEAKING = "SPEAKING"
HUD_STATE_INTERRUPTED = "INTERRUPTED"

HUD_BACKEND_OPENAI = "OPENAI"
HUD_BACKEND_LOCAL = "LOCAL"

HUD_ENABLED = True
hud_q = None
hud_proc = None


def _hud_color_for(state, backend):
    if state == HUD_STATE_IDLE:
        return (35, 35, 40)
    if state == HUD_STATE_RECORDING:
        return (120, 35, 35) if backend == HUD_BACKEND_OPENAI else (35, 120, 55)
    if state == HUD_STATE_THINKING:
        return (35, 65, 120)
    if state == HUD_STATE_SPEAKING:
        return (90, 45, 120)
    if state == HUD_STATE_INTERRUPTED:
        return (140, 120, 35)
    return (35, 35, 40)


def _hud_badge_color(backend):
    if backend == HUD_BACKEND_OPENAI:
        return (220, 80, 80)
    if backend == HUD_BACKEND_LOCAL:
        return (80, 220, 120)
    return (160, 160, 160)


def run_hud(queue):
    """
    Runs in a separate process.
    Receives dict messages:
      {"backend": "OPENAI"/"LOCAL", "state": "...", "status": "...", "memory": int, "flash": bool}
    Special:
      {"cmd": "quit"} exits.
    """
    try:
        import pygame
    except Exception:
        return

    pygame.init()
    screen = pygame.display.set_mode((520, 240))
    pygame.display.set_caption("Lab Partner HUD")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont(None, 34)
    font_body = pygame.font.SysFont(None, 24)
    font_small = pygame.font.SysFont(None, 18)

    backend = HUD_BACKEND_OPENAI
    state = HUD_STATE_IDLE
    memory_turns = 0
    status_line = "Idle / Ready"
    flash_until = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        while True:
            try:
                msg = queue.get_nowait()
            except Exception:
                break

            if isinstance(msg, dict) and msg.get("cmd") == "quit":
                running = False
                break

            if not isinstance(msg, dict):
                continue

            if msg.get("backend") is not None:
                backend = msg["backend"]
            if msg.get("state") is not None:
                state = msg["state"]
            if msg.get("status") is not None:
                status_line = msg["status"]
            if msg.get("memory") is not None:
                memory_turns = int(msg["memory"])
            if msg.get("flash"):
                flash_until = time.time() + 0.35

        screen.fill(_hud_color_for(state, backend))

        panel = pygame.Rect(18, 18, 484, 204)
        pygame.draw.rect(screen, (15, 15, 18), panel, border_radius=16)
        pygame.draw.rect(screen, (55, 55, 65), panel, width=2, border_radius=16)

        title = font_title.render("LAB PARTNER — HUD", True, (235, 235, 240))
        screen.blit(title, (34, 30))

        badge_rect = pygame.Rect(360, 28, 126, 28)
        pygame.draw.rect(screen, _hud_badge_color(backend), badge_rect, border_radius=10)
        screen.blit(font_small.render(backend, True, (15, 15, 18)), (badge_rect.x + 12, badge_rect.y + 7))

        screen.blit(font_body.render("STATE:", True, (200, 200, 210)), (34, 78))
        screen.blit(font_body.render(state, True, (240, 240, 245)), (120, 78))

        screen.blit(font_body.render("MEMORY:", True, (200, 200, 210)), (34, 110))
        screen.blit(font_body.render(f"{memory_turns} turns", True, (240, 240, 245)), (120, 110))

        status_box = pygame.Rect(34, 142, 452, 54)
        pygame.draw.rect(screen, (25, 25, 30), status_box, border_radius=12)
        pygame.draw.rect(screen, (70, 70, 85), status_box, width=2, border_radius=12)
        screen.blit(font_body.render(status_line, True, (235, 235, 245)), (status_box.x + 14, status_box.y + 15))

        screen.blit(font_small.render("Driven by assistant.py events", True, (170, 170, 185)), (34, 202))

        if time.time() < flash_until:
            overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            overlay.fill((255, 220, 80, 90))
            screen.blit(overlay, (0, 0))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def hud_send(*, backend=None, state=None, status=None, memory=None, flash=False):
    global hud_q, HUD_ENABLED
    if not HUD_ENABLED or hud_q is None:
        return
    try:
        hud_q.put_nowait(
            {
                "backend": backend,
                "state": state,
                "status": status,
                "memory": memory,
                "flash": flash,
            }
        )
    except Exception:
        pass


def start_hud():
    global hud_q, hud_proc, HUD_ENABLED
    if not HUD_ENABLED:
        return
    try:
        # IMPORTANT: must be called under __main__ guard
        mp.set_start_method("spawn", force=True)
        hud_q = mp.Queue()
        hud_proc = mp.Process(target=run_hud, args=(hud_q,), daemon=True)
        hud_proc.start()
    except Exception as e:
        print(f"HUD disabled (failed to start): {e}", flush=True)
        HUD_ENABLED = False


# ----------------------------
# GLOBALS (set in main)
# ----------------------------

client = None
cfg = None

DEVICE_PATH = None
SAMPLE_RATE = 16000
LOCAL_TURNS = 6

RED_KEY = None
GREEN_KEY = None
INTERRUPT_KEY = None
ROUTES = None

LOG_ENABLED = True
LOG_PATH = "logs/conversation.jsonl"
MAX_TURNS = 12

TTS_ENGINE = "espeak-ng"
TTS_VOICE = "en-us"
TTS_RATE = 175

persona_text = ""
kb_text = ""

messages = []

# interrupt + recording control
cancel_turn = threading.Event()

# TTS state
tts_proc = None
tts_lock = threading.Lock()
_speak_token = 0
_speak_token_lock = threading.Lock()


def get_turn_count() -> int:
    """
    Returns "turns" as user/assistant pairs, ignoring an initial system message.
    """
    if not messages:
        return 0
    usable = messages[1:] if messages and messages[0].get("role") == "system" else messages
    return len(usable) // 2


# ----------------------------
# AUDIO
# ----------------------------

def record_audio(stop_event: threading.Event):
    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        while not stop_event.is_set():
            sd.sleep(50)

    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)

    return np.concatenate(chunks, axis=0)


def transcribe(audio) -> str:
    if audio is None or getattr(audio, "size", 0) == 0:
        return ""

    # Reject ultra-short audio locally to avoid API 400s
    min_samples = int(0.10 * SAMPLE_RATE)
    if getattr(audio, "shape", (0,))[0] < min_samples:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            try:
                resp = client.audio.transcriptions.create(
                    model=cfg["models"]["stt"],
                    file=f,
                )
            except BadRequestError as e:
                # Common: "Audio file is too short"
                msg = str(e)
                if "audio_too_short" in msg or "too short" in msg:
                    return ""
                raise
        return (resp.text or "").strip()
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


# ----------------------------
# CHAT
# ----------------------------

def _trim():
    global messages
    messages = messages[-MAX_TURNS:]


def chat_openai(text: str, model: str) -> str:
    messages.append({"role": "user", "content": text})
    _trim()

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    reply = (resp.choices[0].message.content or "").strip()

    messages.append({"role": "assistant", "content": reply})
    _trim()
    return reply


def chat_ollama(text: str, model: str) -> str:
    """
    Uses Ollama /api/chat so persona/KB can be sent as a real system message.
    """
    messages.append({"role": "user", "content": text})
    _trim()

    hud_send(backend=HUD_BACKEND_LOCAL, state=HUD_STATE_THINKING, status="Thinking…", memory=get_turn_count())
    print("Thinking (local)...", flush=True)

    local_msgs = []

    system_parts = []
    if persona_text:
        system_parts.append("PERSONA:\n" + persona_text)
    if kb_text:
        system_parts.append("KNOWLEDGE BASE:\n" + kb_text)
    system_context = "\n\n".join(system_parts).strip()
    if system_context:
        local_msgs.append({"role": "system", "content": system_context})

    # Add recent convo (skip system from global list; we already injected above)
    for m in messages[-LOCAL_TURNS:]:
        if m.get("role") == "system":
            continue
        local_msgs.append({"role": m["role"], "content": m["content"]})

    r = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": model,
            "messages": local_msgs,
            "stream": False,
        },
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    reply = ((data.get("message") or {}).get("content") or "").strip()

    messages.append({"role": "assistant", "content": reply})
    _trim()
    return reply


def generate(text: str, route: dict):
    backend = route["backend"]
    if backend == "openai":
        return chat_openai(text, route["chat_model"]), "openai"
    if backend == "ollama":
        return chat_ollama(text, route["ollama_model"]), "ollama"
    raise ValueError(f"Unknown backend: {backend}")


# ----------------------------
# INTERRUPTIBLE SPEECH
# ----------------------------

def _stop_tts_only():
    """Stop TTS without cancelling the whole turn."""
    global tts_proc
    with tts_lock:
        if tts_proc and tts_proc.poll() is None:
            try:
                tts_proc.terminate()
                try:
                    tts_proc.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    tts_proc.kill()
            except Exception:
                pass


def speak(text: str):
    """Start TTS; if already speaking, stop and replace. Also updates HUD back to IDLE when done."""
    global tts_proc, _speak_token
    if not text:
        return

    # bump token for this "generation" of speech
    with _speak_token_lock:
        _speak_token += 1
        my_token = _speak_token

    with tts_lock:
        if tts_proc and tts_proc.poll() is None:
            try:
                tts_proc.terminate()
                try:
                    tts_proc.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    tts_proc.kill()
            except Exception:
                pass

        tts_proc = subprocess.Popen(
            [TTS_ENGINE, "-v", TTS_VOICE, "-s", str(TTS_RATE), text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def watcher(proc, token):
        proc.wait()
        # only the latest speech should flip HUD back to idle
        with _speak_token_lock:
            if token != _speak_token:
                return
        if cancel_turn.is_set():
            # if interrupted mid-speech, leave the last state update to interrupt()
            return
        hud_send(state=HUD_STATE_IDLE, status="Idle / Ready", memory=get_turn_count())

    threading.Thread(target=watcher, args=(tts_proc, my_token), daemon=True).start()


def interrupt():
    """Cancel current turn and stop any active TTS."""
    cancel_turn.set()
    _stop_tts_only()
    print("interrupted", flush=True)
    hud_send(state=HUD_STATE_INTERRUPTED, status="Interrupted!", flash=True, memory=get_turn_count())


# ----------------------------
# MAIN
# ----------------------------

def main():
    global client, cfg
    global DEVICE_PATH, SAMPLE_RATE, LOCAL_TURNS
    global RED_KEY, GREEN_KEY, INTERRUPT_KEY, ROUTES
    global LOG_ENABLED, LOG_PATH, MAX_TURNS
    global TTS_ENGINE, TTS_VOICE, TTS_RATE
    global HUD_ENABLED
    global persona_text, kb_text, messages

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

    HUD_ENABLED = bool(cfg.get("hud", {}).get("enabled", True))

    # Load persona + KB
    persona_text = read_text_file("persona.txt")
    kb_text = read_text_file("knowledge_base.txt")

    system_parts = []
    if persona_text:
        system_parts.append("PERSONA:\n" + persona_text)
    if kb_text:
        system_parts.append("KNOWLEDGE BASE:\n" + kb_text)
    system_context = "\n\n".join(system_parts).strip()

    print("==============================================")
    print(" Lab Partner — Dual PTT + Interrupt + HUD")
    print("==============================================")
    print("Red = OpenAI | Green = Local | Yellow = Interrupt")
    print("Ctrl+C to quit.\n")

    if persona_text:
        print(f"Loaded persona.txt: {len(persona_text)} chars", flush=True)
    if kb_text:
        print(f"Loaded knowledge_base.txt: {len(kb_text)} chars", flush=True)

    messages = []
    if system_context:
        messages.append({"role": "system", "content": system_context})

    # Start HUD (spawn-safe)
    start_hud()
    hud_send(backend=HUD_BACKEND_OPENAI, state=HUD_STATE_IDLE, status="Idle / Ready", memory=get_turn_count())

    device = InputDevice(DEVICE_PATH)

    is_recording = False
    stop_event = None
    active_route = None
    record_thread_local = None
    audio_buf_local = None

    try:
        for event in device.read_loop():
            if event.type != ecodes.EV_KEY:
                continue

            # Interrupt (press)
            if event.code == INTERRUPT_KEY and event.value == 1:
                interrupt()
                continue

            # PTT keys (red/green)
            if event.code in (RED_KEY, GREEN_KEY):
                # PRESS
                if event.value == 1:
                    # IMPORTANT: stop any speech so you don't record the assistant's own voice
                    _stop_tts_only()

                    cancel_turn.clear()
                    is_recording = True
                    stop_event = threading.Event()

                    is_red = (event.code == RED_KEY)
                    active_route = ROUTES["red"] if is_red else ROUTES["green"]

                    label = "openai" if is_red else "local"
                    print(f"recording ({label})...", flush=True)

                    hud_send(
                        backend=HUD_BACKEND_OPENAI if is_red else HUD_BACKEND_LOCAL,
                        state=HUD_STATE_RECORDING,
                        status=f"Recording ({label})…",
                        memory=get_turn_count(),
                    )

                    def runner():
                        nonlocal audio_buf_local
                        audio_buf_local = record_audio(stop_event)

                    record_thread_local = threading.Thread(target=runner, daemon=True)
                    record_thread_local.start()

                # RELEASE
                elif event.value == 0 and is_recording:
                    is_recording = False

                    if stop_event:
                        stop_event.set()

                    if record_thread_local:
                        record_thread_local.join(timeout=2.0)

                    text = transcribe(audio_buf_local)
                    if not text:
                        hud_send(state=HUD_STATE_IDLE, status="Idle / Ready", memory=get_turn_count())
                        continue

                    print("\nYou:", text, flush=True)

                    # show thinking in HUD early for whichever backend is active
                    hud_send(
                        backend=HUD_BACKEND_OPENAI if active_route["backend"] == "openai" else HUD_BACKEND_LOCAL,
                        state=HUD_STATE_THINKING,
                        status="Thinking…",
                        memory=get_turn_count(),
                    )

                    reply, backend = generate(text, active_route)

                    if LOG_ENABLED:
                        append_jsonl(LOG_PATH, {"ts": time.time(), "user": text, "assistant": reply, "backend": backend})

                    # If user hit interrupt during thinking, ignore the reply
                    if cancel_turn.is_set():
                        print("(cancelled turn — ignoring reply)", flush=True)
                        hud_send(state=HUD_STATE_IDLE, status="Cancelled / Ready", memory=get_turn_count())
                        continue

                    print(f"Assistant ({backend}): {reply}\n", flush=True)

                    hud_send(
                        backend=HUD_BACKEND_OPENAI if backend == "openai" else HUD_BACKEND_LOCAL,
                        state=HUD_STATE_SPEAKING,
                        status="Speaking…",
                        memory=get_turn_count(),
                    )

                    speak(reply)

    except KeyboardInterrupt:
        print("\nExiting...", flush=True)
    finally:
        if HUD_ENABLED and hud_q is not None:
            try:
                hud_q.put_nowait({"cmd": "quit"})
            except Exception:
                pass
                

if __name__ == "__main__":
    main()