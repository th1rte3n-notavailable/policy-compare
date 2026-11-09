"""GLFW-backed keyboard state for command input.

The custom viewer owns the window and forwards GLFW key events here.  Because
the demo no longer uses MuJoCo's built-in viewer, none of MuJoCo's keyboard
shortcuts (Space, C, F, P, WASD camera, ...) are active.
"""

from __future__ import annotations

from collections import deque

import glfw

# Keys used by the demo.  The viewer forwards the raw GLFW key code.
_KEY_NAMES: dict[int, str] = {
    glfw.KEY_W: "w",
    glfw.KEY_A: "a",
    glfw.KEY_S: "s",
    glfw.KEY_D: "d",
    glfw.KEY_Q: "q",
    glfw.KEY_E: "e",
    glfw.KEY_R: "r",
    glfw.KEY_C: "c",
    glfw.KEY_P: "p",
    glfw.KEY_F: "f",
    glfw.KEY_X: "x",
    glfw.KEY_SPACE: "space",
    glfw.KEY_ESCAPE: "escape",
}


class KeyboardInput:
    """Held keys plus a queue of discrete key presses."""

    def __init__(self) -> None:
        self._pressed: set[str] = set()
        self._events: deque[str] = deque()

    def on_key(self, key: int, action: int) -> None:
        """Forward a GLFW key event (called from the window key callback)."""
        name = _KEY_NAMES.get(key)
        if name is None:
            return
        if action == glfw.PRESS:
            if name not in self._pressed:
                self._pressed.add(name)
                self._events.append(name)
        elif action == glfw.RELEASE:
            self._pressed.discard(name)
        # glfw.REPEAT is intentionally ignored: held state is polled by the
        # main loop for smooth, joystick-like ramps.

    def clear(self) -> None:
        """Forget all held keys (e.g. when the window loses focus)."""
        self._pressed.clear()

    def consume_events(self) -> list[str]:
        events = list(self._events)
        self._events.clear()
        return events

    def is_pressed(self, name: str) -> bool:
        return name in self._pressed