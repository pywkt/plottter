"""_AnimationMixin — stroke-order animation helpers for CanvasWidget."""
from __future__ import annotations


class _AnimationMixin:
    """Mixin providing animation playback methods for CanvasWidget.

    Must not inherit from QObject.
    """

    def toggle_animation(self) -> None:
        """Toggle animation play/pause. Starts from beginning if not in animation mode."""
        if not self._anim_mode:
            self._rebuild_anim_paths()
            if not self._anim_all_paths:
                return  # nothing to animate
            self._anim_current_path = 0
            self._anim_current_point = 0
            self._anim_mode = True
        if self._anim_playing:
            self._pause_animation()
        else:
            if self._anim_current_path >= len(self._anim_all_paths):
                # Rewind to start
                self._anim_current_path = 0
                self._anim_current_point = 0
            self._play_animation()

    def step_anim_forward(self) -> None:
        """Advance animation by one complete path."""
        if not self._anim_mode:
            self._rebuild_anim_paths()
            if not self._anim_all_paths:
                return
            self._anim_current_path = 0
            self._anim_current_point = 0
            self._anim_mode = True
        if self._anim_current_path < len(self._anim_all_paths):
            self._anim_current_path += 1
            self._anim_current_point = 0
        self._emit_anim_state()
        self.update()

    def step_anim_backward(self) -> None:
        """Go back to the start of the previous path."""
        if not self._anim_mode:
            return
        if self._anim_current_path > 0:
            self._anim_current_path -= 1
            self._anim_current_point = 0
        self._emit_anim_state()
        self.update()

    def seek_animation(self, path_idx: int) -> None:
        """Jump to a specific path index."""
        if not self._anim_mode:
            self._rebuild_anim_paths()
            self._anim_mode = True
        self._anim_current_path = max(0, min(path_idx, len(self._anim_all_paths)))
        self._anim_current_point = 0
        self._emit_anim_state()
        self.update()

    def set_anim_speed(self, speed: float) -> None:
        self._anim_speed = max(0.1, min(10.0, speed))

    def _rebuild_anim_paths(self) -> None:
        """Collect all visible paths for animation, in layer/path order."""
        self._anim_all_paths = []
        for layer in self._controller.current_project.layers:
            if not layer.visible:
                continue
            for polyline in layer.paths:
                if len(polyline) >= 2:
                    self._anim_all_paths.append(
                        (layer.color, layer.opacity, list(polyline))
                    )

    def _play_animation(self) -> None:
        self._anim_playing = True
        self._anim_timer.start()
        self._emit_anim_state()

    def _pause_animation(self) -> None:
        self._anim_playing = False
        self._anim_timer.stop()
        self._emit_anim_state()

    def _reset_animation(self) -> None:
        """Exit animation mode and return to normal rendering."""
        self._anim_playing = False
        self._anim_timer.stop()
        self._anim_mode = False
        self._anim_all_paths = []
        self._anim_current_path = 0
        self._anim_current_point = 0
        self._emit_anim_state()

    def _emit_anim_state(self) -> None:
        self.anim_state_changed.emit(
            self._anim_playing,
            self._anim_current_path,
            len(self._anim_all_paths),
        )

    def _anim_tick(self) -> None:
        """Advance animation state on each timer tick.

        Advancement is distance-based: the pen moves ~80 mm/s (default plotter
        speed) along the path per real-time second, scaled by ``_anim_speed``.
        At 1× speed and a 50 ms tick the budget is 80 × 0.05 = 4 mm per tick,
        so sparse paths (long segments) and dense paths (many short segments)
        animate at the same physical rate.
        """
        if not self._anim_mode or not self._anim_playing:
            return

        _PLOTTER_SPEED_MM_S = 80.0
        tick_s = self.ANIM_TIMER_INTERVAL_MS / 1000.0
        distance_budget = _PLOTTER_SPEED_MM_S * self._anim_speed * tick_s
        changed_path = False
        completed = False

        while distance_budget > 0:
            if self._anim_current_path >= len(self._anim_all_paths):
                self._pause_animation()
                completed = True
                break

            current_polyline = self._anim_all_paths[self._anim_current_path][2]
            next_pt = self._anim_current_point + 1

            if next_pt >= len(current_polyline):
                # Finished this path; advance to the next one
                self._anim_current_path += 1
                self._anim_current_point = 0
                changed_path = True
                continue

            p0 = current_polyline[self._anim_current_point]
            p1 = current_polyline[next_pt]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            seg_dist = (dx * dx + dy * dy) ** 0.5
            # Use at least 1 µm so degenerate (duplicate) points are consumed
            # without causing an infinite loop.
            distance_budget -= max(seg_dist, 1e-3)
            self._anim_current_point = next_pt

        if changed_path and not completed:
            self._emit_anim_state()
        self.update()
