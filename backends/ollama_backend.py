# backends/ollama_backend.py
import requests


def chat_ollama(messages, text: str, model: str, url: str, local_turns: int = 6) -> str:
    """
    Mutates messages in-place by appending user + assistant turns.
    Uses Ollama /api/chat with proper role messages, preserving system persona.
    Returns assistant reply.
    """

    # Grab system prompt if present
    system_text = ""
    if messages and messages[0].get("role") == "system":
        system_text = (messages[0].get("content") or "").strip()

    # Build recent history WITHOUT duplicating the system prompt
    recent = [m for m in messages if m.get("role") != "system"][-(local_turns * 2):]

    # Construct chat payload for /api/chat
    chat_msgs = []
    if system_text:
        chat_msgs.append({"role": "system", "content": system_text})

    chat_msgs.extend(recent)
    chat_msgs.append({"role": "user", "content": text})

    print("Thinking (local)...", flush=True)

    r = requests.post(
        url,
        json={
            "model": model,
            "messages": chat_msgs,
            "stream": False,
        },
        timeout=180,
    )
    r.raise_for_status()

    reply = (r.json().get("message", {}).get("content") or "").strip()

    # Now mutate canonical history (the shared messages list)
    messages.append({"role": "user", "content": text})
    messages.append({"role": "assistant", "content": reply})

    return reply