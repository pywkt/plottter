"""Phase 147.2 — End-to-end multi-layer integration for MapGenerator.

Drives a fixture "fetch" (injecting a pre-built MapData into the panel) then
calls generate through the SettingsPanel in offscreen mode.

Invariants verified:
- After generation: project has one map layer per enabled, non-empty category.
- All generated map layers share a single ``_generator_run_id``.
- A second Generate call replaces (not appends) the prior layers.
"""

from __future__ import annotations

import pytest

from plottter.models import Canvas, Layer, Project


# ---------------------------------------------------------------------------
# Fixtures helpers
# ---------------------------------------------------------------------------


def _make_map_data():
    """Return a MapData fixture with parks (area) and roads_major (line) data."""
    from plottter.osm.types import MapData, MapFeature

    # A rectangular park near Paris (explicitly closed: first == last).
    park_coords = [
        (48.850, 2.340),
        (48.870, 2.340),
        (48.870, 2.370),
        (48.850, 2.370),
        (48.850, 2.340),
    ]
    road_coords = [(48.855, 2.345), (48.865, 2.360)]

    return MapData(
        location="Paris, France",
        center=(48.860, 2.355),
        bbox=(48.845, 2.335, 48.875, 2.380),
        features={
            "parks": [
                MapFeature(
                    tags={"leisure": "park"},
                    coords=park_coords,
                    is_area=True,
                )
            ],
            "roads_major": [
                MapFeature(
                    tags={"highway": "primary"},
                    coords=road_coords,
                    is_area=False,
                )
            ],
        },
    )


def _make_project():
    canvas = Canvas.from_preset("A4", margin=10.0)
    proj = Project(name="MapIntegTest", canvas=canvas)
    layer = Layer(name="Layer 1", color="#000000")
    proj.add_layer(layer)
    return proj, layer.id


def _build_panel(controller, qtbot):
    from plottter.gui.settings_panel import SettingsPanel

    panel = SettingsPanel(controller)
    qtbot.addWidget(panel)
    panel.show()
    panel.on_mode_changed("Map")
    return panel


# ---------------------------------------------------------------------------
# Base params used in generate_layers() calls
# ---------------------------------------------------------------------------

_BASE_PARAMS = {
    "include_water": False,
    "include_parks": True,
    "include_buildings": False,
    "include_waterways": False,
    "include_roads": True,   # enables both roads_minor + roads_major
    "include_rail": False,
    "include_coastline": False,
    "include_attribution": False,  # keep layer count predictable
    "area_fill": "none",
    "simplify_mm": 0.0,
    "min_feature_mm": 0.0,
}


def _run_generate_sync(panel, map_data, extra_params=None):
    """Run MapGenerator.generate_layers() synchronously then call the panel handler.

    This mirrors the path that GeneratorWorker takes in production — it calls
    ``generate_layers()`` in a thread, then emits ``layers_finished``; here we
    shortcut the thread so tests are deterministic and fast.
    """
    from plottter.generators.map_generator import MapGenerator

    gen = panel._generator
    assert isinstance(gen, MapGenerator), (
        f"Expected MapGenerator in Map mode, got {type(gen).__name__}"
    )

    params = {**_BASE_PARAMS, "_map_data": map_data}
    if extra_params:
        params.update(extra_params)

    canvas = panel._controller.current_project.canvas
    specs = gen.generate_layers(params, canvas)
    panel._on_multilayer_generation_finished(specs)
    return specs


def _tagged_layers(controller):
    """Return all project layers carrying a ``_generator_run_id`` tag."""
    return [
        layer
        for layer in controller.current_project.layers
        if isinstance(layer.generator_info, dict)
        and "_generator_run_id" in layer.generator_info
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMapMultiLayerGeneration:
    """First Generate: project gains N map layers all sharing one run id."""

    def test_layers_created_after_generate(self, qtbot):
        proj, lid = _make_project()
        from plottter.gui.project_controller import ProjectController

        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid)
        panel = _build_panel(ctrl, qtbot)

        map_data = _make_map_data()
        panel._map_data = map_data  # simulate a completed fetch

        panel._pending_multilayer_regen_run_id = None
        specs = _run_generate_sync(panel, map_data)

        assert specs, "generate_layers() must return at least one LayerSpec"
        tagged = _tagged_layers(ctrl)
        assert len(tagged) == len(specs), (
            f"Project must have exactly one Layer per LayerSpec "
            f"(expected {len(specs)}, got {len(tagged)})"
        )

    def test_one_layer_per_enabled_nonempty_category(self, qtbot):
        """Each enabled, non-empty category produces exactly one layer."""
        proj, lid = _make_project()
        from plottter.gui.project_controller import ProjectController

        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid)
        panel = _build_panel(ctrl, qtbot)

        map_data = _make_map_data()
        panel._map_data = map_data
        panel._pending_multilayer_regen_run_id = None
        specs = _run_generate_sync(panel, map_data)

        # Fixture has parks + roads_major; include_roads enables roads_minor too,
        # but the fixture contains no roads_minor data so that category is absent.
        # With attribution disabled we expect exactly 2 layers: Parks + Roads (major).
        category_names = {s.name for s in specs}
        assert "Parks" in category_names, f"Parks layer expected; got {category_names}"
        assert "Roads (major)" in category_names, (
            f"Roads (major) layer expected; got {category_names}"
        )

    def test_all_layers_share_one_run_id(self, qtbot):
        """All layers produced by one Generate call carry a single shared run id."""
        proj, lid = _make_project()
        from plottter.gui.project_controller import ProjectController

        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid)
        panel = _build_panel(ctrl, qtbot)

        map_data = _make_map_data()
        panel._map_data = map_data
        panel._pending_multilayer_regen_run_id = None
        specs = _run_generate_sync(panel, map_data)

        assert specs, "Need at least one spec to check run ids"
        tagged = _tagged_layers(ctrl)
        assert tagged, "No tagged layers found after generate"

        run_ids = {layer.generator_info["_generator_run_id"] for layer in tagged}
        assert len(run_ids) == 1, (
            f"All map layers must share exactly one _generator_run_id; "
            f"found {len(run_ids)}: {run_ids}"
        )

    def test_generator_name_tagged(self, qtbot):
        """Each generated layer must carry the MapGenerator's name."""
        proj, lid = _make_project()
        from plottter.gui.project_controller import ProjectController
        from plottter.generators.map_generator import MapGenerator

        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid)
        panel = _build_panel(ctrl, qtbot)

        map_data = _make_map_data()
        panel._map_data = map_data
        panel._pending_multilayer_regen_run_id = None
        _run_generate_sync(panel, map_data)

        expected_name = MapGenerator.name
        tagged = _tagged_layers(ctrl)
        for layer in tagged:
            assert layer.generator_info.get("_generator_name") == expected_name, (
                f"Layer {layer.name!r} has generator_name="
                f"{layer.generator_info.get('_generator_name')!r}; "
                f"expected {expected_name!r}"
            )

    def test_preexisting_plain_layer_untouched(self, qtbot):
        """A first Generate must not remove the pre-existing plain 'Layer 1'."""
        proj, lid = _make_project()
        from plottter.gui.project_controller import ProjectController

        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid)
        panel = _build_panel(ctrl, qtbot)

        map_data = _make_map_data()
        panel._map_data = map_data
        panel._pending_multilayer_regen_run_id = None
        _run_generate_sync(panel, map_data)

        plain_layers = [
            layer
            for layer in ctrl.current_project.layers
            if layer.generator_info is None
        ]
        assert len(plain_layers) >= 1, (
            "Pre-existing plain 'Layer 1' must not be removed by a first Generate"
        )


class TestMapRegenReplaces:
    """Second Generate: replaces prior map layers — does not append."""

    def _do_first_generate(self, panel, map_data):
        """Simulate first Generate and return the shared run_id from that run."""
        panel._pending_multilayer_regen_run_id = None
        _run_generate_sync(panel, map_data)
        ctrl = panel._controller
        tagged = _tagged_layers(ctrl)
        assert tagged, "First generate must produce tagged layers"
        run_id = tagged[0].generator_info["_generator_run_id"]
        assert all(
            l.generator_info["_generator_run_id"] == run_id for l in tagged
        ), "All first-run layers must share the same run id"
        return run_id

    def test_second_generate_replaces_first_layers(self, qtbot):
        """Re-generate replaces old layers; old run_id absent from project."""
        proj, lid = _make_project()
        from plottter.gui.project_controller import ProjectController

        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid)
        panel = _build_panel(ctrl, qtbot)

        map_data = _make_map_data()
        panel._map_data = map_data

        old_run_id = self._do_first_generate(panel, map_data)

        # Second generate — simulate what _on_generate sets before launching the worker
        panel._pending_multilayer_regen_run_id = old_run_id
        _run_generate_sync(panel, map_data)

        tagged = _tagged_layers(ctrl)
        remaining_run_ids = {l.generator_info["_generator_run_id"] for l in tagged}

        assert old_run_id not in remaining_run_ids, (
            "Old run layers must be removed after re-generation "
            f"(old_run_id={old_run_id!r} still present)"
        )

    def test_layer_count_stable_after_regen(self, qtbot):
        """Total tagged layer count must not grow on re-generation."""
        proj, lid = _make_project()
        from plottter.gui.project_controller import ProjectController

        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid)
        panel = _build_panel(ctrl, qtbot)

        map_data = _make_map_data()
        panel._map_data = map_data

        old_run_id = self._do_first_generate(panel, map_data)
        count_after_run1 = len(_tagged_layers(ctrl))

        # Second generate
        panel._pending_multilayer_regen_run_id = old_run_id
        _run_generate_sync(panel, map_data)

        count_after_run2 = len(_tagged_layers(ctrl))
        assert count_after_run2 == count_after_run1, (
            f"Re-generation must replace layers, not append: "
            f"run1={count_after_run1}, run2={count_after_run2}"
        )

    def test_second_run_has_fresh_run_id(self, qtbot):
        """The second Generate must produce a new (distinct) run_id."""
        proj, lid = _make_project()
        from plottter.gui.project_controller import ProjectController

        ctrl = ProjectController(proj)
        ctrl.set_active_layer(lid)
        panel = _build_panel(ctrl, qtbot)

        map_data = _make_map_data()
        panel._map_data = map_data

        old_run_id = self._do_first_generate(panel, map_data)

        panel._pending_multilayer_regen_run_id = old_run_id
        _run_generate_sync(panel, map_data)

        tagged = _tagged_layers(ctrl)
        new_run_id = tagged[0].generator_info["_generator_run_id"]
        assert new_run_id != old_run_id, (
            "Each Generate must produce a fresh UUID run_id"
        )

    def test_prior_run_lookup_across_project(self, qtbot):
        """Prior-run lookup scans the whole project, not just the active layer.

        Regression: if the user is on the original empty 'Layer 1' when they
        click Generate a second time, the prior run must still be found so
        layers are replaced rather than appended.
        """
        proj, original_lid = _make_project()
        from plottter.gui.project_controller import ProjectController

        ctrl = ProjectController(proj)
        ctrl.set_active_layer(original_lid)
        panel = _build_panel(ctrl, qtbot)

        map_data = _make_map_data()
        panel._map_data = map_data

        # First generate — produces tagged layers, active layer stays on Layer 1
        old_run_id = self._do_first_generate(panel, map_data)
        count_after_run1 = len(_tagged_layers(ctrl))

        # Simulate active layer still being the untagged original Layer 1
        ctrl.set_active_layer(original_lid)

        # Simulate the project-wide prior-run lookup that _on_generate performs
        from plottter.generators.map_generator import MapGenerator

        prior_run_id = None
        for proj_layer in ctrl.current_project.layers:
            info = proj_layer.generator_info
            if (
                isinstance(info, dict)
                and info.get("_generator_name") == MapGenerator.name
                and info.get("_generator_run_id")
            ):
                prior_run_id = info["_generator_run_id"]

        assert prior_run_id == old_run_id, (
            "Prior-run lookup must find the previous Map run even when the "
            "active layer is the untagged starter layer"
        )

        # Second generate using that prior run id
        panel._pending_multilayer_regen_run_id = prior_run_id
        _run_generate_sync(panel, map_data)

        count_after_run2 = len(_tagged_layers(ctrl))
        assert count_after_run2 == count_after_run1, (
            "Layer count must not grow when re-generating from an untagged active layer"
        )


# ---------------------------------------------------------------------------
# Helpers for positioned generate tests (phase 151.3)
# ---------------------------------------------------------------------------


def _make_positioned_fixture():
    """Parks in NW corner, road in SE corner — spatially separated for positioning tests.

    The two features are in opposite quadrants of the bbox so that a 3× zoom
    into the NW area makes the SE road fall completely outside the printable
    area (verified analytically in spec §8, phase 149.4 / 151.3).
    """
    from plottter.osm.types import MapData, MapFeature

    # Small closed park polygon in the NW corner of the bbox.
    park_nw = MapFeature(
        tags={"leisure": "park"},
        coords=[
            (48.870, 2.340),
            (48.868, 2.340),
            (48.868, 2.343),
            (48.870, 2.343),
            (48.870, 2.340),
        ],
        is_area=True,
    )
    # Short road segment in the SE corner of the bbox.
    road_se = MapFeature(
        tags={"highway": "primary"},
        coords=[(48.851, 2.372), (48.850, 2.375)],
        is_area=False,
    )
    return MapData(
        location="test-positioned",
        center=(48.860, 2.358),
        bbox=(48.850, 2.340, 48.870, 2.375),
        features={"parks": [park_nw], "roads_major": [road_se]},
    )


def _poly_length(paths) -> float:
    """Total Euclidean length (mm) across all polylines in *paths*."""
    total = 0.0
    for path in paths:
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            total += (dx * dx + dy * dy) ** 0.5
    return total


def _run_generate_direct(map_data, canvas, extra_params=None):
    """Call MapGenerator.generate_layers() directly (no panel, no Qt thread)."""
    from plottter.generators.map_generator import MapGenerator

    params = {**_BASE_PARAMS, "_map_data": map_data}
    if extra_params:
        params.update(extra_params)
    return MapGenerator().generate_layers(params, canvas)


# ---------------------------------------------------------------------------
# Tests — phase 151.3
# ---------------------------------------------------------------------------


class TestMapPositionedGenerate:
    """Phase 151.3 — end-to-end positioned generate with ``_map_view``.

    Spec §8:
    - With a zoomed/panned ``_map_view``: a feature positioned outside the
      printable area is absent; surviving features are scaled larger than the
      fit baseline.
    - Without ``_map_view``: output matches the fit baseline (both categories
      present; regression guard).
    """

    @staticmethod
    def _canvas():
        return Canvas.from_preset("A4", margin=10.0)

    @staticmethod
    def _zoomed_nw_view(map_data, canvas):
        """Return a ``_map_view`` dict zoomed 3× into the NW corner (park area).

        Analytically verified: at 3× fit_scale centred on (48.870°, 2.340°)
        the SE road segment (at lon ≈ 2.372°–2.375°) lies beyond the right
        edge of the viewport after clamp_map_view — it is completely absent
        from the clipped output.
        """
        from plottter.osm.geometry import default_map_view

        all_features = (
            map_data.features.get("parks", [])
            + map_data.features.get("roads_major", [])
        )
        fit_view = default_map_view(all_features, canvas)
        return {
            "center_lat": 48.870,
            "center_lon": 2.340,
            "scale": 3.0 * fit_view["scale"],
        }

    # ------------------------------------------------------------------

    def test_no_view_baseline_has_both_categories(self):
        """Without ``_map_view`` both Parks and Roads (major) appear (fit baseline)."""
        canvas = self._canvas()
        map_data = _make_positioned_fixture()

        specs = _run_generate_direct(map_data, canvas)
        names = {s.name for s in specs}

        assert "Parks" in names, (
            f"Fit baseline must include a Parks layer; got {names}"
        )
        assert "Roads (major)" in names, (
            f"Fit baseline must include a Roads (major) layer; got {names}"
        )

    def test_zoomed_view_drops_distant_feature(self):
        """With view zoomed into the NW, the SE road is absent from output."""
        canvas = self._canvas()
        map_data = _make_positioned_fixture()

        zoomed_view = self._zoomed_nw_view(map_data, canvas)
        specs = _run_generate_direct(
            map_data, canvas, extra_params={"_map_view": zoomed_view}
        )
        names = {s.name for s in specs}

        assert "Parks" in names, (
            f"Parks (NW) must survive in zoomed output; got {names}"
        )
        assert "Roads (major)" not in names, (
            f"Roads (major) (SE) must be absent when zoomed into NW; got {names}"
        )

    def test_zoomed_survivors_scaled_larger_than_fit(self):
        """Surviving paths are physically larger (mm) at 3× zoom vs fit."""
        canvas = self._canvas()
        map_data = _make_positioned_fixture()

        zoomed_view = self._zoomed_nw_view(map_data, canvas)

        fit_specs = _run_generate_direct(map_data, canvas)
        zoomed_specs = _run_generate_direct(
            map_data, canvas, extra_params={"_map_view": zoomed_view}
        )

        fit_parks = next((s for s in fit_specs if s.name == "Parks"), None)
        zoomed_parks = next((s for s in zoomed_specs if s.name == "Parks"), None)

        assert fit_parks is not None, "Parks must be present in fit baseline output"
        assert zoomed_parks is not None, "Parks must survive in zoomed output"

        fit_len = _poly_length(fit_parks.paths)
        zoomed_len = _poly_length(zoomed_parks.paths)

        assert zoomed_len > fit_len, (
            f"Zoomed Parks paths must be physically larger (mm) than the fit "
            f"baseline: fit={fit_len:.3f} mm, zoomed={zoomed_len:.3f} mm"
        )
