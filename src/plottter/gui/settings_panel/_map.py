"""_MapMixin — Map Location group box: location field, extent label, fetch button, status."""

from __future__ import annotations


class _MapMixin:
    """Mixin composed into SettingsPanel for Map-mode location + fetch UI.

    Builds a ``_map_group`` QGroupBox (visible only in Map mode) containing:

    - A ``QLineEdit`` for the location query.
    - A read-only extent summary label (updates when Fetch is clicked).
    - A "Fetch Map Data" button that checks the disk cache first, then starts
      a ``_MapFetchWorker`` if no cached data is available.
    - A status label reporting Idle / Geocoding… / progress / success / error.
    """

    def _build_map_group(self) -> None:
        """Create ``self._map_group`` and add it to ``self._layout``."""
        from PyQt6.QtWidgets import (
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QVBoxLayout,
        )

        # Map-view state (spec §6.1) — initialised to None; set when data arrives.
        self._map_view: dict | None = None

        self._map_group = QGroupBox("Map Location")
        map_layout = QVBoxLayout(self._map_group)

        # Location text field
        self._map_location_edit = QLineEdit()
        self._map_location_edit.setPlaceholderText("City, address, or place\u2026")
        map_layout.addWidget(self._map_location_edit)

        # Extent summary label — updated when Fetch is clicked
        self._map_extent_label = QLabel("Extent: 1.5\u202fkm radius")
        self._map_extent_label.setStyleSheet("color: #aaa; font-size: 11px;")
        map_layout.addWidget(self._map_extent_label)

        # "Fetch Map Data" button
        fetch_row = QHBoxLayout()
        self._map_fetch_btn = QPushButton("Fetch Map Data")
        self._map_fetch_btn.clicked.connect(self._on_fetch_map_clicked)
        fetch_row.addWidget(self._map_fetch_btn)
        fetch_row.addStretch()
        map_layout.addLayout(fetch_row)

        # Status label
        self._map_status_label = QLabel("Idle")
        self._map_status_label.setWordWrap(True)
        map_layout.addWidget(self._map_status_label)

        # Positioning controls — disabled until map data is loaded (spec §6.1)
        pos_row = QHBoxLayout()
        self._map_position_btn = QPushButton("Position Map")
        self._map_position_btn.setCheckable(True)
        self._map_position_btn.setEnabled(False)
        self._map_position_btn.toggled.connect(self._on_position_map_toggled)
        pos_row.addWidget(self._map_position_btn)

        self._map_reset_btn = QPushButton("Reset to fit")
        self._map_reset_btn.setEnabled(False)
        self._map_reset_btn.clicked.connect(self._on_reset_to_fit_clicked)
        pos_row.addWidget(self._map_reset_btn)
        map_layout.addLayout(pos_row)

        self._layout.addWidget(self._map_group)
        self._map_group.setVisible(False)

    # ------------------------------------------------------------------
    # Fetch button handler
    # ------------------------------------------------------------------

    def _on_fetch_map_clicked(self) -> None:
        """Handle 'Fetch Map Data' button: check cache, then start worker."""
        location = self._map_location_edit.text().strip()
        if not location:
            self._map_status_label.setText("Please enter a location.")
            self._map_status_label.setStyleSheet("color: orange;")
            return

        # Read current generator params for cache-key derivation and worker args
        params = self.get_params()
        radius_km = float(params.get("radius_km", 1.5))
        extent_mode = str(params.get("extent_mode", "radius"))
        enabled_cats = self._get_enabled_map_categories(params)

        # Update the extent summary label with the current radius
        self._map_extent_label.setText(f"Extent: {radius_km:.3g}\u202fkm radius")

        # --- Disk cache check (§11) ---
        cached = None
        cache_hit_key: str | None = None
        try:
            from plottter.osm.cache import cache_key as _cache_key
            from plottter.osm.cache import load as _cache_load

            cache_hit_key = _cache_key(location, radius_km, extent_mode, enabled_cats)
            cached = _cache_load(cache_hit_key)
        except Exception:  # noqa: BLE001 — cache errors must never surface
            pass

        if cached is not None:
            self._map_data = cached
            self._init_map_view_from_data(cached)
            n = sum(len(v) for v in cached.features.values())
            self._map_status_label.setText(f"Loaded: {n:,} features (cached)")
            self._map_status_label.setStyleSheet("")
            return

        # --- Start network fetch ---
        self._map_fetch_btn.setEnabled(False)
        self._map_status_label.setText("Geocoding\u2026")
        self._map_status_label.setStyleSheet("")
        # Remember the key so the finished handler can write to cache
        self._map_fetch_cache_key = cache_hit_key

        from PyQt6.QtCore import QSettings

        from plottter.gui.settings_panel.workers import _MapFetchWorker

        settings = QSettings("Plottter", "Plottter")
        endpoint: str = settings.value(
            "map/overpass_endpoint",
            "https://overpass-api.de/api/interpreter",
            type=str,
        )

        worker = _MapFetchWorker(
            location=location,
            radius_km=radius_km,
            extent_mode=extent_mode,
            selectors=enabled_cats,
            endpoint=endpoint,
        )
        worker.progress.connect(self._on_map_fetch_progress)
        worker.finished.connect(self._on_map_fetch_finished)
        worker.error.connect(self._on_map_fetch_error)
        self._map_fetch_worker = worker
        worker.start()

    def _get_enabled_map_categories(self, params: dict) -> list:
        """Return the list of enabled OSM category IDs from current param values."""
        cats: list = []
        if params.get("include_roads", True):
            cats.append("roads_major")
            if params.get("road_detail", "standard") != "major_only":
                cats.append("roads_minor")
        if params.get("include_rail", True):
            cats.append("rail")
        if params.get("include_water", True):
            cats.append("water")
        if params.get("include_waterways", True):
            cats.append("waterways")
        if params.get("include_parks", True):
            cats.append("parks")
        if params.get("include_buildings", False):
            cats.append("buildings")
        if params.get("include_coastline", True):
            cats.append("coastline")
        if params.get("include_place_labels", True):
            cats.append("places")
        return cats

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------

    def _on_map_fetch_progress(self, pct: int) -> None:
        """Update status label with download progress."""
        if pct < 15:
            self._map_status_label.setText("Geocoding\u2026")
        else:
            self._map_status_label.setText(f"Downloading features\u2026 {pct}%")

    def _on_map_fetch_finished(self, map_data: object) -> None:
        """Store MapData, write disk cache, update project metadata, re-enable button."""
        # The `finished` signal is emitted from INSIDE _MapFetchWorker.run(),
        # so when this slot runs (via a queued connection on the main thread)
        # the worker thread is still wrapping up. If we drop the Python ref
        # right now, the QThread destructor fires before run() has actually
        # returned -> "QThread: Destroyed while thread is still running" +
        # SIGABRT. wait() blocks until the thread truly terminates (in
        # practice sub-millisecond, since emit is run()'s last statement).
        worker = self._map_fetch_worker
        self._map_fetch_worker = None
        if worker is not None:
            worker.wait(5000)

        self._map_data = map_data

        # Write to disk cache — best-effort, never crash
        try:
            key = getattr(self, "_map_fetch_cache_key", None)
            if key is not None:
                from plottter.osm.cache import store as _cache_store

                _cache_store(key, map_data)
        except Exception:  # noqa: BLE001
            pass
        self._map_fetch_cache_key = None

        # Persist attribution to project metadata (ODbL requirement §12)
        try:
            if self._controller is not None:
                project = self._controller.current_project
                if project is not None:
                    attribution = getattr(map_data, "attribution", "© OpenStreetMap contributors")
                    project.metadata["map_attribution"] = attribution
        except Exception:  # noqa: BLE001
            pass

        n = sum(len(v) for v in map_data.features.values())
        self._map_status_label.setText(f"Loaded: {n:,} features")
        self._map_status_label.setStyleSheet("")
        self._map_fetch_btn.setEnabled(True)

        # Initialise default map view and enable positioning controls (spec §6.1)
        self._init_map_view_from_data(map_data)

    def _on_map_fetch_error(self, msg: str) -> None:
        """Show error message in red and re-enable the fetch button."""
        # Same lifetime concern as _on_map_fetch_finished — wait for run() to
        # truly return before letting the QThread object be destroyed.
        worker = self._map_fetch_worker
        self._map_fetch_worker = None
        if worker is not None:
            worker.wait(5000)
        self._map_fetch_cache_key = None
        self._map_status_label.setText(f"Error: {msg}")
        self._map_status_label.setStyleSheet("color: red;")
        self._map_fetch_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Map view initialisation (spec §6.1)
    # ------------------------------------------------------------------

    def _init_map_view_from_data(self, map_data: object) -> None:
        """Compute or restore map view from data; enable positioning controls.

        Restores from ``project.metadata["map_view"]`` if present (spec §6.3),
        otherwise falls back to ``default_map_view`` (fit-to-canvas).
        """
        try:
            all_features = [f for flist in map_data.features.values() for f in flist]
            if not all_features:
                return
            project = self._controller.current_project if self._controller is not None else None
            if project is None:
                return

            # Restore persisted view if available (spec §6.3)
            persisted = project.metadata.get("map_view")
            if persisted and isinstance(persisted, dict) and all(
                k in persisted for k in ("center_lat", "center_lon", "scale")
            ):
                self._map_view = dict(persisted)
            else:
                from plottter.osm.geometry import default_map_view

                self._map_view = default_map_view(all_features, project.canvas)
        except Exception:  # noqa: BLE001 — positioning is best-effort
            return

        self._map_position_btn.setEnabled(True)
        self._map_reset_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Positioning button handlers (spec §6.1)
    # ------------------------------------------------------------------

    def _on_position_map_toggled(self, checked: bool) -> None:
        """Toggle map-positioning interactive mode on the canvas."""
        if self._canvas_ref is None:
            self._map_position_btn.setChecked(False)
            return
        self._canvas_ref.set_map_position_active(checked)
        if checked and self._map_data is not None:
            try:
                from plottter.osm.geometry import data_bounds

                all_features = [f for flist in self._map_data.features.values() for f in flist]
                bounds = data_bounds(all_features)
                self._canvas_ref.set_map_preview_data(self._map_data, bounds)
                if self._map_view is not None:
                    self._canvas_ref.update_map_view(self._map_view)
            except Exception:  # noqa: BLE001
                pass

    def _on_reset_to_fit_clicked(self) -> None:
        """Reset the map view to the fit-to-canvas default."""
        if self._map_data is None:
            return
        try:
            project = self._controller.current_project if self._controller is not None else None
            if project is None:
                return
            from plottter.osm.geometry import default_map_view

            all_features = [f for flist in self._map_data.features.values() for f in flist]
            self._map_view = default_map_view(all_features, project.canvas)
        except Exception:  # noqa: BLE001
            return

        if self._canvas_ref is not None:
            self._canvas_ref.update_map_view(self._map_view)

    # ------------------------------------------------------------------
    # Canvas → panel sync (spec §6.2)
    # ------------------------------------------------------------------

    def _on_canvas_map_view_changed(self, lat: float, lon: float, scale: float) -> None:
        """Store the updated map view emitted by the canvas and persist it."""
        self._map_view = {"center_lat": lat, "center_lon": lon, "scale": scale}
        # Persist to project metadata (spec §6.3) — mirrors camera persistence
        try:
            if self._controller is not None:
                project = self._controller.current_project
                if project is not None:
                    project.metadata["map_view"] = dict(self._map_view)
        except Exception:  # noqa: BLE001 — persistence is best-effort
            pass
