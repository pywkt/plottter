"""Background QThread workers for MainWindow operations."""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from plottter.models.path import Polyline


class _WeldWorker(QThread):
    """QThread that runs weld_overlapping_paths on a layer's paths."""

    finished = pyqtSignal(list, int, int)  # (new_paths, before_count, after_count)
    progress = pyqtSignal(int, int)        # (current_index, total)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        paths: list[Polyline],
        tolerance_mm: float = 0.1,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths
        self._tolerance_mm = tolerance_mm
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            from plottter.processing.weld import weld_overlapping_paths

            before_count = len(self._paths)
            new_paths = weld_overlapping_paths(
                self._paths,
                tolerance_mm=self._tolerance_mm,
                cancelled_callback=lambda: self._cancelled,
                progress_callback=lambda cur, tot: self.progress.emit(cur, tot),
            )
            if self._cancelled:
                self.cancelled.emit()
            else:
                after_count = len(new_paths)
                self.finished.emit(new_paths, before_count, after_count)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class _OptimizeWorker(QThread):
    """QThread that runs the full path optimization pipeline on a layer's paths."""

    finished = pyqtSignal(list, float, float, int, int)  # (new_paths, before_travel, after_travel, before_lifts, after_lifts)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)  # 0-100 within-layer progress

    def __init__(
        self,
        paths: list[Polyline],
        generator_info: dict | None = None,
        run_weld: bool = False,
        weld_tolerance: float = 0.1,
        run_simplify: bool = True,
        simplify_tolerance: float = 0.1,
        run_filter: bool = True,
        filter_min_length: float = 0.5,
        run_clip: bool = True,
        clip_bounds: tuple[float, float, float, float] | None = None,
        run_merge: bool = True,
        merge_threshold: float = 0.5,
        run_join: bool = False,
        join_threshold: float = 0.1,
        run_2opt: bool = True,
        run_3opt: bool = False,
        run_or_opt: bool = True,
        num_starts: int = 5,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths

        # Auto-enable Join for map layers — the graph-aware Eulerian-walk
        # algorithm reduces pen lifts ≥30% beyond what plain Merge achieves on
        # road networks (benchmark: 54 → 24 lifts on a 220-segment city grid).
        # Does not affect the user's saved preference for non-map layers.
        _is_map = (
            isinstance(generator_info, dict)
            and generator_info.get("_generator_name") == "Map"
        )
        if _is_map and not run_join:
            run_join = True

        self._run_weld = run_weld
        self._weld_tolerance = weld_tolerance
        self._run_simplify = run_simplify
        self._simplify_tolerance = simplify_tolerance
        self._run_filter = run_filter
        self._filter_min_length = filter_min_length
        self._run_clip = run_clip
        self._clip_bounds = clip_bounds
        self._run_merge = run_merge
        self._merge_threshold = merge_threshold
        self._run_join = run_join
        self._join_threshold = join_threshold
        self._run_2opt = run_2opt
        self._run_3opt = run_3opt
        self._run_or_opt = run_or_opt
        self._num_starts = num_starts
        self._cancelled = False

    def request_stop(self) -> None:
        """Request cancellation.  The worker will stop at the next safe checkpoint."""
        self._cancelled = True

    def run(self) -> None:
        try:
            from plottter.processing import run_optimization_pipeline

            settings = {
                "run_weld": self._run_weld,
                "weld_tolerance": self._weld_tolerance,
                "run_simplify": self._run_simplify,
                "simplify_tolerance": self._simplify_tolerance,
                "run_filter": self._run_filter,
                "filter_min_length": self._filter_min_length,
                "run_clip": self._run_clip,
                "run_merge": self._run_merge,
                "merge_threshold": self._merge_threshold,
                "run_join": self._run_join,
                "join_threshold": self._join_threshold,
                "run_2opt": self._run_2opt,
                "run_3opt": self._run_3opt,
                "run_or_opt": self._run_or_opt,
                "num_starts": self._num_starts,
            }
            # generator_info is consumed inside the pipeline (force-on Join
            # for Map layers). We've already mutated self._run_join in
            # __init__ for the same reason, so pass None here to avoid
            # double-toggling.
            result = run_optimization_pipeline(
                self._paths,
                settings=settings,
                clip_bounds=self._clip_bounds,
                generator_info=None,
                progress_callback=lambda v: self.progress.emit(v),
                cancelled=lambda: self._cancelled,
            )
            self.finished.emit(
                result.paths,
                result.before_travel,
                result.after_travel,
                result.before_lifts,
                result.after_lifts,
            )
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class _RemoteOptimizeWorker(QThread):
    """Offload the optimize pipeline to a remote machine over SSH.

    Spawns ``ssh <host> plottter --optimize`` and pipes the same JSON
    payload the local CLI consumes. Progress lines on the remote's
    stderr are parsed and re-emitted as ``progress`` signals so the
    existing progress dialog works unchanged.

    Cancellation kills the SSH subprocess (which propagates SIGTERM to
    the remote plottter), so a half-finished optimize doesn't keep
    burning remote CPU after the user clicks Cancel.
    """

    finished = pyqtSignal(list, float, float, int, int)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(
        self,
        paths: list[Polyline],
        host: str,
        remote_command: str = "plottter",
        generator_info: dict | None = None,
        run_weld: bool = False,
        weld_tolerance: float = 0.1,
        run_simplify: bool = True,
        simplify_tolerance: float = 0.1,
        run_filter: bool = True,
        filter_min_length: float = 0.5,
        run_clip: bool = True,
        clip_bounds: tuple[float, float, float, float] | None = None,
        run_merge: bool = True,
        merge_threshold: float = 0.5,
        run_join: bool = False,
        join_threshold: float = 0.1,
        run_2opt: bool = True,
        run_3opt: bool = False,
        run_or_opt: bool = True,
        num_starts: int = 5,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths
        self._host = host
        self._remote_command = (remote_command or "plottter").strip()
        self._generator_info = generator_info
        self._settings = {
            "run_weld": run_weld,
            "weld_tolerance": weld_tolerance,
            "run_simplify": run_simplify,
            "simplify_tolerance": simplify_tolerance,
            "run_filter": run_filter,
            "filter_min_length": filter_min_length,
            "run_clip": run_clip,
            "run_merge": run_merge,
            "merge_threshold": merge_threshold,
            "run_join": run_join,
            "join_threshold": join_threshold,
            "run_2opt": run_2opt,
            "run_3opt": run_3opt,
            "run_or_opt": run_or_opt,
            "num_starts": num_starts,
        }
        self._clip_bounds = clip_bounds
        self._proc = None
        self._cancelled = False

    def request_stop(self) -> None:
        self._cancelled = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass

    def run(self) -> None:
        import json
        import shlex
        import subprocess
        import threading

        if not self._host:
            self.error.emit("Remote host not configured.")
            return

        payload = {
            "paths": [[[x, y] for x, y in poly] for poly in self._paths],
            "settings": self._settings,
            "clip_bounds": list(self._clip_bounds) if self._clip_bounds else None,
            "generator_info": self._generator_info,
        }

        # Pass through ~/.ssh/config (Host blocks, ControlMaster, agent auth,
        # IdentityFile, etc.) — that's why the user's existing SSH setup
        # "just works" for this tool. The remote command splits on shlex so
        # users can override it with an absolute venv path (e.g.
        # ``/home/.../.venv/bin/plottter``) or extra flags like
        # ``bash -lc 'plottter'``.
        remote = shlex.split(self._remote_command) if self._remote_command else ["plottter"]
        cmd = ["ssh", self._host, *remote, "--optimize"]

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            self.error.emit(
                "'ssh' not found on PATH. Install an SSH client to use "
                "remote optimization."
            )
            return
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"Failed to spawn ssh: {exc}")
            return

        # We need *streaming* progress from stderr while we also collect a
        # single payload on stdout. ``subprocess.communicate()`` would drain
        # both pipes itself, racing any pump thread we attach to stderr —
        # so drive the three FDs by hand instead.
        captured_stderr: list[str] = []

        def _pump_stderr() -> None:
            stream = self._proc.stderr if self._proc is not None else None
            if stream is None:
                return
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    captured_stderr.append(line)
                    continue
                if isinstance(msg, dict) and "progress" in msg:
                    try:
                        self.progress.emit(int(msg["progress"]))
                    except (TypeError, ValueError):
                        pass
                else:
                    captured_stderr.append(line)

        pump = threading.Thread(target=_pump_stderr, daemon=True)
        pump.start()

        # Write input then close stdin so the remote sees EOF.
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write(json.dumps(payload).encode("utf-8"))
            self._proc.stdin.close()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"SSH stdin write failed: {exc}")
            self._proc.kill()
            return

        # Read stdout to EOF on this thread (pump owns stderr).
        try:
            assert self._proc.stdout is not None
            stdout = self._proc.stdout.read()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"SSH stdout read failed: {exc}")
            self._proc.kill()
            return

        self._proc.wait()
        pump.join(timeout=2.0)

        if self._cancelled:
            return

        if self._proc.returncode != 0:
            tail = "\n".join(captured_stderr[-10:]) if captured_stderr else ""
            cmd_str = " ".join(shlex.quote(c) for c in cmd)
            self.error.emit(
                f"Remote optimize failed (exit {self._proc.returncode}).\n"
                f"Command: {cmd_str}\n"
                f"Remote stderr:\n{tail or '(empty)'}"
            )
            return

        try:
            result = json.loads(stdout.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            self.error.emit(f"Could not parse remote response: {exc}")
            return

        try:
            new_paths = [
                [(float(x), float(y)) for x, y in poly]
                for poly in result["paths"]
            ]
            self.finished.emit(
                new_paths,
                float(result["before_travel"]),
                float(result["after_travel"]),
                int(result["before_lifts"]),
                int(result["after_lifts"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            self.error.emit(f"Remote response missing fields: {exc}")


class _BrushWorker(QThread):
    """QThread that runs apply_brush on a layer's paths."""

    finished = pyqtSignal(list)  # (new_paths,)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)  # 0-100

    def __init__(
        self,
        paths: list[Polyline],
        brush_type: str,
        params: dict,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths
        self._brush_type = brush_type
        self._params = params

    def run(self) -> None:
        try:
            from plottter.processing.brush import apply_brush
            self.progress.emit(10)
            result = apply_brush(self._paths, self._brush_type, self._params)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class _TaperWorker(QThread):
    """QThread that runs taper_paths on a layer's paths."""

    finished = pyqtSignal(list)  # (new_paths,)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)  # 0-100

    def __init__(
        self,
        paths: list[Polyline],
        params: dict,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths
        self._params = params

    def run(self) -> None:
        try:
            from plottter.processing.taper import taper_paths
            self.progress.emit(10)
            result = taper_paths(self._paths, **self._params)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class _OffsetWorker(QThread):
    """QThread that runs offset_paths on a layer's paths."""

    finished = pyqtSignal(list)  # (new_paths,)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)  # 0-100

    def __init__(
        self,
        paths: list[Polyline],
        params: dict,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._paths = paths
        self._params = params

    def run(self) -> None:
        try:
            from plottter.processing.offset import offset_paths
            self.progress.emit(10)
            result = offset_paths(self._paths, **self._params)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
