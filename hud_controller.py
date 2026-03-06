# hud_controller.py
import threading
from typing import Optional


class HudController:
    def __init__(self):
        self.enabled = False
        self._queue = None
        self._hud_state = None
        self._hud_shutdown = None

        try:
            import queue as _queue
            from hud import run_hud, hud_state, hud_shutdown

            self._queue = _queue.Queue()
            threading.Thread(target=run_hud, args=(self._queue,), daemon=True).start()

            self._hud_state = hud_state
            self._hud_shutdown = hud_shutdown
            self.enabled = True

            print("[HUD] started", flush=True)

        except Exception as e:
            self.enabled = False
            self._queue = None
            print(f"[HUD] disabled (failed to start): {type(e).__name__}: {e}", flush=True)

    def set(self, state_name: str, extra: Optional[str] = None):
        if not self.enabled:
            return
        try:
            self._hud_state(self._queue, state_name, extra)
        except Exception:
            pass

    def shutdown(self):
        if not self.enabled:
            return
        try:
            self._hud_shutdown(self._queue)
        except Exception:
            pass
