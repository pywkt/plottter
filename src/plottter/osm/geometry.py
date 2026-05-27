"""Web Mercator projection and fit-to-canvas geometry for the Map generator.

See spec §6 in specs/map-generator.md for the canonical design.
"""

import math

_LAT_CLAMP = 85.05112877980659  # degrees — avoids log(0) singularity at ±90°


def mercator(lat: float, lon: float) -> tuple[float, float]:
    """Project (lat, lon) to Web Mercator (x, y) with unit sphere radius.

    Latitude is clamped to ±85.05° to avoid the pole singularity.
    The caller rescales the output in fit-to-canvas (§6.2).

    Args:
        lat: Latitude in decimal degrees (WGS84).
        lon: Longitude in decimal degrees (WGS84).

    Returns:
        (x, y) in radians — x increases eastward, y increases northward.
    """
    lat = max(-_LAT_CLAMP, min(_LAT_CLAMP, lat))
    x = math.radians(lon)
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


from dataclasses import dataclass


def inverse_mercator(x: float, y: float) -> tuple[float, float]:
    """Inverse of mercator(): (x, y) in radians-units → (lat, lon) in degrees.

    Args:
        x: Mercator x coordinate (radians), increases eastward.
        y: Mercator y coordinate (radians), increases northward.

    Returns:
        (lat, lon) in decimal degrees (WGS84).
    """
    lon = math.degrees(x)
    lat = math.degrees(2 * math.atan(math.exp(y)) - math.pi / 2)
    return lat, lon


@dataclass
class FitTransform:
    """Affine transform mapping Mercator (x, y) to canvas mm coordinates.

    Apply as::

        px, py = mercator(lat, lon)
        canvas_x = transform.x_origin + px * transform.scale
        canvas_y = transform.y_origin - py * transform.scale

    The y-axis is flipped so that north (larger Mercator y) maps to a
    smaller canvas y (toward the top of the page).
    """

    scale: float
    x_origin: float  # canvas_x = x_origin + px * scale
    y_origin: float  # canvas_y = y_origin - py * scale  (y-flipped)


def fit_transform(features: list, canvas) -> FitTransform:
    """Derive an affine transform fitting all features inside canvas.drawing_area().

    Projects every coordinate of every feature via Web Mercator, computes the
    projected bounding box, then derives a uniform scale and centering offsets
    so the full map fits inside the printable area with the aspect ratio
    preserved (spec §6.2).  The y-axis is flipped so north is up.

    Args:
        features: Iterable of MapFeature objects (each has .coords as a list
                  of (lat, lon) pairs in decimal degrees).
        canvas:   Canvas dataclass whose drawing_area() returns
                  (left, top, right, bottom) in mm.

    Returns:
        A FitTransform instance.

    Raises:
        ValueError: If no coordinates are found across all features.
    """
    xs: list[float] = []
    ys: list[float] = []
    for feature in features:
        for lat, lon in feature.coords:
            px, py = mercator(lat, lon)
            xs.append(px)
            ys.append(py)

    if not xs:
        raise ValueError(
            "No coordinates found in features; cannot compute fit transform."
        )

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    left, top, right, bottom = canvas.drawing_area()
    pw = right - left
    ph = bottom - top

    span_x = maxx - minx
    span_y = maxy - miny

    if span_x == 0.0 and span_y == 0.0:
        scale = 1.0
    elif span_x == 0.0:
        scale = ph / span_y
    elif span_y == 0.0:
        scale = pw / span_x
    else:
        scale = min(pw / span_x, ph / span_y)

    # Centering margins so the drawing is centred inside the printable area.
    cx = (pw - span_x * scale) / 2.0
    cy = (ph - span_y * scale) / 2.0

    # Derive origins so the application formula is simply:
    #   canvas_x = x_origin + px * scale
    #   canvas_y = y_origin - py * scale
    x_origin = left + cx - minx * scale
    y_origin = top + cy + maxy * scale

    return FitTransform(scale=scale, x_origin=x_origin, y_origin=y_origin)


def view_transform(
    center_lat: float,
    center_lon: float,
    scale: float,
    canvas,
) -> FitTransform:
    """Build a FitTransform placing (center_lat, center_lon) at the centre of canvas.drawing_area().

    Args:
        center_lat: Latitude of the geographic point to place at the canvas centre.
        center_lon: Longitude of the geographic point to place at the canvas centre.
        scale:      mm per Mercator unit (same quantity as FitTransform.scale).
        canvas:     Canvas dataclass whose drawing_area() returns
                    (left, top, right, bottom) in mm.

    Returns:
        A FitTransform that maps the given geographic point to the printable-area
        centre at the requested scale, with north up (y-flipped).
    """
    mcx, mcy = mercator(center_lat, center_lon)
    left, top, right, bottom = canvas.drawing_area()
    ccx = (left + right) / 2
    ccy = (top + bottom) / 2
    x_origin = ccx - mcx * scale
    y_origin = ccy + mcy * scale  # y-flipped (north up), matches MG §6.2
    return FitTransform(scale=scale, x_origin=x_origin, y_origin=y_origin)


def default_map_view(features: list, canvas) -> dict:
    """Return the {center_lat, center_lon, scale} view equivalent to fit_transform.

    The returned dict can be passed to view_transform() to reproduce a FitTransform
    that frames the data identically to fit_transform() (same projected coordinates
    within floating-point precision).

    The geographic centre is the inverse-projected Mercator midpoint of the data
    bbox; the scale is taken directly from fit_transform (spec §3.3).

    Args:
        features: Iterable of MapFeature objects (each has .coords as a list
                  of (lat, lon) pairs in decimal degrees).
        canvas:   Canvas dataclass whose drawing_area() returns
                  (left, top, right, bottom) in mm.

    Returns:
        A dict with keys ``center_lat``, ``center_lon``, ``scale``.

    Raises:
        ValueError: If no coordinates are found across all features.
    """
    xs: list[float] = []
    ys: list[float] = []
    for feature in features:
        for lat, lon in feature.coords:
            px, py = mercator(lat, lon)
            xs.append(px)
            ys.append(py)

    if not xs:
        raise ValueError(
            "No coordinates found in features; cannot compute default map view."
        )

    ft = fit_transform(features, canvas)

    mcx = (min(xs) + max(xs)) / 2.0
    mcy = (min(ys) + max(ys)) / 2.0
    center_lat, center_lon = inverse_mercator(mcx, mcy)

    return {"center_lat": center_lat, "center_lon": center_lon, "scale": ft.scale}


def data_bounds(features: list) -> tuple[float, float, float, float]:
    """Geographic bbox (min_lat, min_lon, max_lat, max_lon) over all feature coords.

    Args:
        features: Iterable of MapFeature objects (each has .coords as a list
                  of (lat, lon) pairs in decimal degrees).

    Returns:
        (min_lat, min_lon, max_lat, max_lon) in decimal degrees.

    Raises:
        ValueError: If no coordinates are found across all features.
    """
    lats: list[float] = []
    lons: list[float] = []
    for feature in features:
        for lat, lon in feature.coords:
            lats.append(lat)
            lons.append(lon)
    if not lats:
        raise ValueError(
            "No coordinates found in features; cannot compute data bounds."
        )
    return min(lats), min(lons), max(lats), max(lons)


def clamp_map_view(view: dict, features: list, canvas) -> dict:
    """Clamp a view so the printable-area viewport stays within the fetched data
    extent and scale never drops below fit (no whitespace, no zoom-out past fit).

    Clamp rules (spec §3.4):
    - scale = max(view["scale"], fit_scale) where fit_scale comes from
      fit_transform(features, canvas).
    - The viewport's geographic half-extents at the clamped scale are
      printable_width / (2 * scale) and printable_height / (2 * scale)
      in Mercator units.
    - center_lat/lon are clamped so that the viewport rectangle lies inside
      the data's Mercator bounding box.
    - If data is smaller than the viewport in a dimension (only possible on
      the letterbox axis at fit scale), the centre is placed at the data
      midpoint in that dimension.

    Args:
        view:     dict with keys center_lat, center_lon, scale.
        features: Iterable of MapFeature objects.
        canvas:   Canvas dataclass whose drawing_area() returns
                  (left, top, right, bottom) in mm.

    Returns:
        A new dict with clamped center_lat, center_lon, scale.

    Raises:
        ValueError: If no coordinates are found across all features.
    """
    fit_scale = fit_transform(features, canvas).scale
    scale = max(view["scale"], fit_scale)

    left, top, right, bottom = canvas.drawing_area()
    pw = right - left   # printable width in mm
    ph = bottom - top   # printable height in mm

    half_w = (pw / 2.0) / scale   # viewport half-width in Mercator units
    half_h = (ph / 2.0) / scale   # viewport half-height in Mercator units

    # Mercator bounding box of the data.
    xs: list[float] = []
    ys: list[float] = []
    for feature in features:
        for lat, lon in feature.coords:
            px, py = mercator(lat, lon)
            xs.append(px)
            ys.append(py)

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    # Current centre in Mercator.
    mcx, mcy = mercator(view["center_lat"], view["center_lon"])

    # Clamp x: if data span is narrower than viewport width, centre on data;
    # otherwise push the edges to stay within data bounds.
    if maxx - minx <= 2.0 * half_w:
        mcx = (minx + maxx) / 2.0
    else:
        mcx = max(minx + half_w, min(maxx - half_w, mcx))

    # Clamp y: same logic (Mercator y increases northward).
    if maxy - miny <= 2.0 * half_h:
        mcy = (miny + maxy) / 2.0
    else:
        mcy = max(miny + half_h, min(maxy - half_h, mcy))

    center_lat, center_lon = inverse_mercator(mcx, mcy)
    return {"center_lat": center_lat, "center_lon": center_lon, "scale": scale}


def project_feature(feature, transform: FitTransform) -> list[tuple[float, float]]:
    """Project a MapFeature's coordinates to canvas mm using the given transform.

    Args:
        feature:   MapFeature with .coords as [(lat, lon), ...].
        transform: FitTransform returned by fit_transform().

    Returns:
        List of (x_mm, y_mm) canvas coordinates in mm.
    """
    result: list[tuple[float, float]] = []
    for lat, lon in feature.coords:
        px, py = mercator(lat, lon)
        canvas_x = transform.x_origin + px * transform.scale
        canvas_y = transform.y_origin - py * transform.scale
        result.append((canvas_x, canvas_y))
    return result


def clip_lines(
    polylines: list[list[tuple[float, float]]],
    bbox_rect_mm: tuple[float, float, float, float],
    min_len_mm: float,
) -> list[list[tuple[float, float]]]:
    """Clip polylines to a bounding box, dropping short fragments.

    Uses shapely ``LineString.intersection(box)`` per spec §6.3.  Multi-part
    intersection results are split into individual polylines; any segment
    shorter than *min_len_mm* is discarded.

    Args:
        polylines:    Each entry is a list of (x_mm, y_mm) canvas coordinates.
        bbox_rect_mm: (left, top, right, bottom) clipping rectangle in mm.
        min_len_mm:   Minimum arc length (mm) to keep; shorter fragments are
                      dropped.

    Returns:
        Clipped polylines, each as a list of (x_mm, y_mm) tuples.
    """
    from shapely.geometry import LineString, MultiLineString, GeometryCollection
    from shapely.geometry import box as shapely_box

    left, top, right, bottom = bbox_rect_mm
    clip_box = shapely_box(left, top, right, bottom)

    result: list[list[tuple[float, float]]] = []
    for coords in polylines:
        if len(coords) < 2:
            continue
        clipped = LineString(coords).intersection(clip_box)
        if clipped.is_empty:
            continue
        if isinstance(clipped, LineString):
            geoms = [clipped]
        elif isinstance(clipped, (MultiLineString, GeometryCollection)):
            geoms = [g for g in clipped.geoms if isinstance(g, LineString)]
        else:
            continue
        for g in geoms:
            if g.length >= min_len_mm:
                result.append(list(g.coords))
    return result


def clip_polygons(
    polylines: list[list[tuple[float, float]]],
    bbox_rect_mm: tuple[float, float, float, float],
    min_len_mm: float,
) -> list[list[tuple[float, float]]]:
    """Clip polygon outlines to a bounding box, dropping small results.

    Uses shapely ``Polygon.intersection(box)`` per spec §6.3.  Multi-part
    results are split into individual rings; rings whose perimeter is shorter
    than *min_len_mm* are discarded.

    Args:
        polylines:    Each entry is a polygon ring as (x_mm, y_mm) tuples.
                      The ring may be open or closed (first == last).
        bbox_rect_mm: (left, top, right, bottom) clipping rectangle in mm.
        min_len_mm:   Minimum perimeter (mm) to keep; shorter rings are
                      dropped.

    Returns:
        Clipped polygon outlines, each as a **closed** list of (x_mm, y_mm)
        tuples (shapely exterior coords include the repeated closing vertex).
    """
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
    from shapely.geometry import box as shapely_box

    left, top, right, bottom = bbox_rect_mm
    clip_box = shapely_box(left, top, right, bottom)

    result: list[list[tuple[float, float]]] = []
    for coords in polylines:
        if len(coords) < 3:
            continue
        polygon = Polygon(coords)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        clipped = polygon.intersection(clip_box)
        if clipped.is_empty:
            continue
        if isinstance(clipped, Polygon):
            geoms = [clipped]
        elif isinstance(clipped, (MultiPolygon, GeometryCollection)):
            geoms = [g for g in clipped.geoms if isinstance(g, Polygon)]
        else:
            continue
        for g in geoms:
            if g.length >= min_len_mm:
                result.append(list(g.exterior.coords))
    return result


def assemble(
    feature: "MapFeature",  # type: ignore[name-defined]  # forward ref; imported below
) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
    """Assemble a MapFeature into geometric rings or polylines per spec §6.4.

    - **Open way** (``is_area=False``): returns a single open polyline; the
      caller must not close it.
    - **Closed way / polygon** (``is_area=True``): returns a single closed
      ring where ``ring[0] == ring[-1]``.  If the stored ``coords`` are not
      yet closed the first point is appended automatically.
    - **Relation (multipolygon)**: outer ring is handled identically to a
      closed way.  Inner-member rings (``role=inner``) stored in
      ``feature.inner_coords`` are returned as *inner_holes* for fill
      clipping (§7).

    Args:
        feature: A :class:`~plottter.osm.types.MapFeature`.
                 ``feature.coords`` is the (lat, lon) sequence of the outer
                 ring or open way.  ``feature.inner_coords`` (may be empty)
                 holds inner rings from multipolygon relation members.

    Returns:
        ``(rings_or_lines, inner_holes)`` — both are lists of coordinate
        lists (each coordinate list is a sequence of (lat, lon) tuples).
        For open ways ``inner_holes`` is always ``[]``.
    """
    coords = list(feature.coords)

    if not feature.is_area:
        # Open way → polyline returned as-is; first != last is expected.
        return ([coords], [])

    # Closed way or relation outer ring — ensure ring closure.
    if len(coords) >= 2 and coords[0] != coords[-1]:
        coords = coords + [coords[0]]

    inner_holes: list[list[tuple[float, float]]] = [
        list(ring) for ring in feature.inner_coords
    ]
    return ([coords], inner_holes)
