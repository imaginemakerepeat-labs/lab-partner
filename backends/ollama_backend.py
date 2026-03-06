import requests


def chat_ollama(messages, cancel_turn, user_text: str, model: str, ollama_url: str, local_turns: int) -> str:
    cancel_turn.clear()

    sys_msg = messages[0] if (messages and messages[0]["role"] == "system") else None
    recent = [m for m in messages if m["role"] != "system"][-(local_turns * 2):]
    local_msgs = ([sys_msg] if sys_msg else []) + recent + [{"role": "user", "content": user_text}]

    payload = {"model": model, "messages": local_msgs, "stream": False}

    try:
        r = requests.post(ollama_url, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        out = (data.get("message", {}) or {}).get("content", "")
    except Exception as e:
        out = f"(ollama error) {e}"

    if cancel_turn.is_set():
        return ""

    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": (out or "").strip()})
    return (out or "").strip()