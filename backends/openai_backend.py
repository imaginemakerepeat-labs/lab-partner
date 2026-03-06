# backends/openai_backend.py

def chat_openai(client, messages, text: str, model: str) -> str:
    """
    Mutates messages in-place by appending user + assistant turns.
    Returns assistant reply.
    """
    messages.append({"role": "user", "content": text})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    reply = (resp.choices[0].message.content or "").strip()
    messages.append({"role": "assistant", "content": reply})
    return reply
