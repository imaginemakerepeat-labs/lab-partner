	# memory.py
from typing import List, Dict, Optional


class ChatMemory:
    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        self.messages: List[Dict] = []

    def set_system(self, system_text: str):
        system_text = (system_text or "").strip()
        if not system_text:
            return

        # replace existing system message if present
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = {"role": "system", "content": system_text}
        else:
            self.messages.insert(0, {"role": "system", "content": system_text})

        self.trim()

    def trim(self):
        # Preserve system prompt as messages[0] if present
        if self.messages and self.messages[0].get("role") == "system":
            sys_msg = self.messages[0]
            tail = self.messages[1:]
            tail = tail[-max(0, self.max_turns - 1):]
            self.messages = [sys_msg] + tail
        else:
            self.messages = self.messages[-self.max_turns:]

    def append(self, role: str, content: str, **extra):
        msg = {"role": role, "content": content}
        msg.update(extra)
        self.messages.append(msg)
        self.trim()

    def list(self) -> List[Dict]:
        return self.messages
