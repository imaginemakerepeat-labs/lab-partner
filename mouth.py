import socket
import time
import threading


MOUTH_DEBUG = True
MOUTH_DEBUG_EVERY = 1


class MouthController:
    def __init__(self, enabled: bool, ip: str, port: int):
        self.enabled = enabled
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        self.sent = 0

    def send(self, cmd: str, why: str = "") -> None:
        if not self.enabled:
            return

        self.seq += 1
        self.sent += 1
        payload = f"{self.seq}|{time.time():.3f}|{cmd}"

        try:
            self.sock.sendto(payload.encode("utf-8"), (self.ip, self.port))
            if MOUTH_DEBUG and (self.sent % MOUTH_DEBUG_EVERY == 0):
                print(f"[MOUTH->] seq={self.seq} cmd={cmd} why={why}", flush=True)
        except Exception as e:
            print(f"[MOUTH!!] send failed seq={self.seq} cmd={cmd} err={e}", flush=True)

    def ticker_loop(self, stop_evt: threading.Event, cancel_turn: threading.Event) -> None:
        print("[MOUTH] loop start", flush=True)
        cycle = ["open", "wide", "open", "close"]
        idx = 0

        try:
            while not stop_evt.is_set() and not cancel_turn.is_set():
                cmd = cycle[idx % len(cycle)]
                self.send(cmd, why="loop")
                idx += 1
                time.sleep(0.08)
        except Exception as e:
            print(f"[MOUTH!!] loop exception: {e}", flush=True)

        self.send("close", why="loop_end")
        self.send("clear", why="loop_end")
        print("[MOUTH] loop end", flush=True)