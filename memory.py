"""
Memory helpers for Lab Partner
Handles conversation history trimming
"""

def trim_messages(messages, max_turns: int):
    if not messages:
        return messages

    sys_msg = messages[0] if messages[0]["role"] == "system" else None
    rest = messages[1:] if sys_msg else messages
    rest = rest[-(max_turns * 2):]

    if sys_msg:
        return [sys_msg] + rest

    return rest