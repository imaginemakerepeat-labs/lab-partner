def chat_openai(client, messages, cancel_turn, user_text: str, model: str) -> str:
    cancel_turn.clear()
    messages.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(model=model, messages=messages)

    if cancel_turn.is_set():
        return ""

    out = (resp.choices[0].message.content or "").strip()
    messages.append({"role": "assistant", "content": out})
    return out