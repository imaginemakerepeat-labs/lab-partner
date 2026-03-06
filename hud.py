# hud.py
import time
import pygame

STATE_IDLE = "IDLE"
STATE_RECORDING = "RECORDING"
STATE_THINKING = "THINKING"
STATE_SPEAKING = "SPEAKING"
STATE_INTERRUPTED = "INTERRUPTED"

BACKEND_OPENAI = "OPENAI"
BACKEND_LOCAL = "LOCAL"


def color_for(state, backend):
    if state == STATE_RECORDING:
        return (220, 60, 60)
    if state == STATE_THINKING:
        return (240, 180, 60)
    if state == STATE_SPEAKING:
        return (60, 200, 120)
    if state == STATE_INTERRUPTED:
        return (200, 80, 200)
    return (200, 200, 200)


def hud_state(q, state, extra=None):
    """Push HUD update messages into the queue."""
    if q is None:
        return
    try:
        q.put_nowait({"state": state, "backend": extra, "ts": time.time()})
    except Exception:
        pass


def hud_shutdown(q):
    """Tell the HUD loop to exit."""
    if q is None:
        return
    try:
        q.put_nowait({"cmd": "quit"})
    except Exception:
        pass


def run_hud(q):
    """
    Pygame HUD loop.
    Exits cleanly when it receives {"cmd":"quit"} from the queue or the window is closed.
    """

    # Try to init video safely
    try:
        pygame.init()
        pygame.display.init()
    except Exception:
        # Can't init video (headless), just return
        return

    if not pygame.display.get_init():
        return

    width, height = 520, 200
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Lab Partner HUD")

    font_big = pygame.font.SysFont("Arial", 64)
    font_med = pygame.font.SysFont("Arial", 28)
    font_small = pygame.font.SysFont("Arial", 22)

    state = STATE_IDLE
    backend = ""
    last_ts = 0.0

    clock = pygame.time.Clock()
    running = True

    while running:
        # Drain queue (keep newest)
        if q is not None:
            try:
                while True:
                    msg = q.get_nowait()
                    if isinstance(msg, dict) and msg.get("cmd") == "quit":
                        running = False
                        break
                    state = msg.get("state", state)
                    backend = msg.get("backend", backend)
                    last_ts = msg.get("ts", last_ts)
            except Exception:
                pass

        # If pygame got de-initialized externally, bail
        if not pygame.get_init() or not pygame.display.get_init():
            break

        # Handle window events safely
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        except pygame.error:
            # video system not initialized
            break

        # Draw background
        screen.fill((15, 15, 18))

        color = color_for(state, backend)

        # State
        text_state = font_big.render(str(state), True, color)
        screen.blit(text_state, (30, 40))

        # Backend
        if backend:
            text_backend = font_med.render(str(backend).upper(), True, (200, 200, 200))
            screen.blit(text_backend, (32, 120))

        # Age
        if last_ts:
            age = time.time() - last_ts
            text_age = font_small.render(f"updated {age:.1f}s ago", True, (130, 130, 130))
            screen.blit(text_age, (32, 160))

        pygame.display.flip()
        clock.tick(30)

    try:
        pygame.display.quit()
    except Exception:
        pass
    try:
        pygame.quit()
    except Exception:
        pass