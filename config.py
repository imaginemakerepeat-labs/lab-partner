# config.py
import json
from pathlib import Path
from evdev import ecodes

# ----------------------------
# DEFAULT CONFIG
# ----------------------------

DEFAULT_CONFIG = {
    "device_path": "/dev/input/by-id/usb-0079_USB_Gamepad-event-joystick",
    "sample_rate": 16000,
    "models": {"stt": "whisper-1", "chat": "gpt-4o"},
    "tts": {"engine": "espeak-ng", "voice": "en-us", "rate": 175},
    "logging": {
        "enabled": True,
        "path": "logs/conversation.jsonl",
        "max_turns_in_memory": 12,
    },
    "context": {"local_turns": 6},
    "buttons": {
        "red_ptt": "BTN_THUMB",
        "green_ptt": "BTN_TOP",
        "interrupt": "BTN_PINKIE",
    },
    "routes": {
        "red": {"backend": "openai", "chat_model": "gpt-4o"},
        "green": {"backend": "ollama", "ollama_model": "qwen2.5:1.5b-instruct"},
    },
    "ollama": {"url": "http://127.0.0.1:11434/api/chat"},
    "maynard": {"enabled": True, "ip": "10.0.0.4", "port": 9000},
    "prompt_files": {"persona": "persona.txt", "knowledge_base": "knowledge_base.txt"},
}

# ----------------------------
# CONFIG HELPERS
# ----------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _ecode(name: str) -> int:
    if not hasattr(ecodes, name):
        raise ValueError(f"Unknown evdev code name: {name}")
    return getattr(ecodes, name)


def load_config(path: str = "config.json") -> dict:
    cfg = dict(DEFAULT_CONFIG)
    p = Path(path)

    if p.exists():
        try:
            cfg = _deep_merge(cfg, json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"⚠️ Failed to load {path}: {e}", flush=True)

    cfg["red_key"] = _ecode(cfg["buttons"]["red_ptt"])
    cfg["green_key"] = _ecode(cfg["buttons"]["green_ptt"])
    cfg["interrupt_key"] = _ecode(cfg["buttons"]["interrupt"])
    return cfg


def append_jsonl(path: str, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def read_text_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return ""