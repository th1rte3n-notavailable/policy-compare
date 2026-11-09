"""A small GLFW + MuJoCo renderer with fully custom key handling.

The built-in ``mujoco.viewer`` binds many keys (Space, C, F, P, WASD camera,
...) that collide with the policy command controls.  This viewer creates its
own GLFW window and renders with ``mjv_updateScene`` / ``mjr_render``, so the
only keyboard bindings are the ones implemented in this project.

Mouse controls: left-drag orbits, right-drag pans, scroll wheel zooms.
"""

from __future__ import annotations

import glfw
import mujoco
import numpy as np

from .keyboard import KeyboardInput

FONT_BIG = mujoco.mjtFont.mjFONT_BIG
FONT_NORMAL = mujoco.mjtFont.mjFONT_NORMAL
GRID_TOP_LEFT = mujoco.mjtGridPos.mjGRID_TOPLEFT
GRID_BOTTOM_LEFT = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT


class GlfwViewer:
    """Minimal interactive viewer for one MuJoCo model."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        keyboard: KeyboardInput | None = None,
        *,
        width: int = 1280,
        height: int = 720,
        title: str = "policy-compare",
    ) -> None:
        self.model = model
        self.data = data
        self.keyboard = keyboard
        self.window = None
        self._drag: str | None = None
        self._last_cursor: tuple[float, float] | None = None

        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")

        glfw.window_hint(glfw.SAMPLES, 4)
        self.window = glfw.create_window(
            width, height, title, None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create a GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.lookat[:] = [0.0, 0.0, 0.35]
        self.cam.distance = 2.0
        self.cam.azimuth = 135.0
        self.cam.elevation = -18.0

        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0  # hide collision geoms
        self.option.sitegroup[:] = 0  # hide sensor/equality sites
        self.option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1

        self.perturb = mujoco.MjvPerturb()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.context = mujoco.MjrContext(
            model, mujoco.mjtFontScale.mjFONTSCALE_150
        )
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_window_focus_callback(self.window, self._on_focus)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    # ------------------------------------------------------------------
    # Input callbacks
    # ------------------------------------------------------------------

    def _on_key(self, _window, key: int, _scancode: int, action: int, _mods: int) -> None:
        if self.keyboard is not None:
            self.keyboard.on_key(key, action)

    def _on_focus(self, _window, focused: bool) -> None:
        if focused:
            return
        self._drag = None
        self._last_cursor = None
        if self.keyboard is not None:
            # No key-release events arrive while unfocused; avoid stuck keys.
            self.keyboard.clear()

    def _on_mouse_button(self, _window, button: int, action: int, _mods: int) -> None:
        if action == glfw.PRESS:
            if button == glfw.MOUSE_BUTTON_LEFT:
                self._drag = "rotate"
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self._drag = "pan"
            else:
                return
            self._last_cursor = glfw.get_cursor_pos(self.window)
        elif action == glfw.RELEASE:
            if (button == glfw.MOUSE_BUTTON_LEFT and self._drag == "rotate") or (
                button == glfw.MOUSE_BUTTON_RIGHT and self._drag == "pan"
            ):
                self._drag = None
                self._last_cursor = None

    def _on_cursor(self, _window, xpos: float, ypos: float) -> None:
        if self._drag is None:
            self._last_cursor = (xpos, ypos)
            return
        if self._last_cursor is None:
            self._last_cursor = (xpos, ypos)
            return
        dx = xpos - self._last_cursor[0]
        dy = ypos - self._last_cursor[1]
        self._last_cursor = (xpos, ypos)

        width, height = glfw.get_window_size(self.window)
        if width <= 0 or height <= 0:
            return
        if self._drag == "rotate":
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                dx / width,
                dy / height,
                self.cam,
            )
        else:
            # Pan in the screen plane: horizontal drag moves along the camera
            # right axis, vertical drag along the camera up axis.
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_MOVE_H,
                dx / width,
                0.0,
                self.cam,
            )
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_MOVE_V,
                0.0,
                dy / height,
                self.cam,
            )

    def _on_scroll(self, _window, _xoffset: float, yoffset: float) -> None:
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            yoffset * 0.1,
            self.cam,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def poll(self) -> None:
        glfw.poll_events()

    def is_running(self) -> bool:
        return self.window is not None and not glfw.window_should_close(self.window)

    def set_lookat(self, position: np.ndarray) -> None:
        self.cam.lookat[:] = position

    def _viewport(self) -> mujoco.MjrRect | None:
        width, height = glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            return None
        return mujoco.MjrRect(0, 0, width, height)

    def render(
        self,
        status_text: str = "",
        help_text: str = "",
        *,
        swap: bool = True,
    ) -> None:
        viewport = self._viewport()
        if viewport is None:
            return
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        if status_text:
            mujoco.mjr_overlay(
                FONT_BIG,
                GRID_TOP_LEFT,
                viewport,
                status_text,
                "",
                self.context,
            )
        if help_text:
            mujoco.mjr_overlay(
                FONT_NORMAL,
                GRID_BOTTOM_LEFT,
                viewport,
                help_text,
                "",
                self.context,
            )
        if swap:
            glfw.swap_buffers(self.window)

    def read_pixels(self) -> np.ndarray:
        """Read the current back buffer as a top-down RGB uint8 array."""
        viewport = self._viewport()
        if viewport is None:
            raise RuntimeError("No drawable framebuffer")
        rgb = np.empty((viewport.height, viewport.width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, self.context)
        return rgb[::-1].copy()

    def close(self) -> None:
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()