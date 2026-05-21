"""Generator ABC and Parameter/Preset dataclasses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from plottter.models import Canvas, Polyline


@dataclass
class Parameter:
    """Base class for generator parameters."""

    name: str
    label: str
    randomizable: bool = True
    visible_when: dict[str, list[str]] | None = None
    description: str = ""


@dataclass
class FloatParam(Parameter):
    min: float = 0.0
    max: float = 100.0
    step: float = 0.1
    default: float = 0.0


@dataclass
class IntParam(Parameter):
    min: int = 0
    max: int = 100
    step: int = 1
    default: int = 0


@dataclass
class ExpressionParam(Parameter):
    default: str = ""
    variables: list[str] = field(default_factory=list)


@dataclass
class ChoiceParam(Parameter):
    choices: list[str] = field(default_factory=list)
    default: str = ""
    choice_descriptions: dict[str, str] | None = None


@dataclass
class BoolParam(Parameter):
    default: bool = False


@dataclass
class ColorParam(Parameter):
    default: str = "#000000"


@dataclass
class ImageParam(Parameter):
    default: str = ""


@dataclass
class FileParam(Parameter):
    """A file-path parameter rendered as a QLineEdit + Browse button.

    Parameters
    ----------
    default:    Initial file path (may be empty).
    filter:     Qt file-dialog filter string, e.g. ``"Mesh Files (*.stl *.obj);;All Files (*)"``
    """

    default: str = ""
    filter: str = "All Files (*)"


@dataclass
class StringParam(Parameter):
    """A free-text parameter rendered as a text input widget.

    Parameters
    ----------
    default:    Initial text value.
    multiline:  If True, render as a multi-line text editor instead of a
                single-line QLineEdit.  Newlines in the string are passed
                through to the generator as-is.
    """

    default: str = ""
    multiline: bool = False


@dataclass
class FontParam(Parameter):
    """A font-selection parameter rendered as a :class:`FontPicker` widget.

    The value passed to the generator is a string containing the absolute
    path to the selected ``.ttf`` or ``.otf`` file (same as what the old
    ``system_font_path`` StringParam used to hold).  An empty string means
    no font is selected.

    Parameters
    ----------
    default:    Initial font file path (may be empty).
    """

    default: str = ""
    randomizable: bool = False


@dataclass
class Preset:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class LayerSpec:
    """Specification for a single layer emitted by a multi-layer generator.

    Used when ``Generator.emits_multiple_layers`` is ``True``.  The generator
    returns a list of ``LayerSpec`` objects from ``generate_layers()`` and the
    GUI creates one ``Layer`` per spec.
    """

    name: str
    color: str
    paths: list[Polyline]


class Generator(ABC):
    """Abstract base class for all art generators."""

    name: str = ""
    category: str = ""
    #: Set to True in generators that accept a preprocessed image via
    #: ``params["_source_image"]`` (numpy array injected by the settings panel).
    uses_source_image: bool = False
    #: Set to True in generators that emit multiple named layers via
    #: ``generate_layers()`` instead of a single ``list[Polyline]`` via
    #: ``generate()``.
    emits_multiple_layers: bool = False

    @abstractmethod
    def get_parameters(self) -> list[Parameter]:
        """Return the list of parameters for this generator."""

    @abstractmethod
    def generate(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[Polyline]:
        """Run the generator and return a list of polylines (coordinates in mm)."""

    def generate_layers(
        self,
        params: dict[str, Any],
        canvas: Canvas,
        progress_callback: Any = None,
        cancelled_callback: Any = None,
    ) -> list[LayerSpec]:
        """Run the generator and return a list of LayerSpec objects.

        Override this method (and set ``emits_multiple_layers = True``) in
        generators that need to produce multiple named, coloured layers in a
        single generation pass.  The default implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    @abstractmethod
    def get_presets(self) -> list[Preset]:
        """Return the list of named presets for this generator."""

    @staticmethod
    def get_post_processing_parameters() -> list[Parameter]:
        """Return post-processing (brush effect) parameters shared by all generators.

        These are rendered in a separate "Post-Processing" group in the settings
        panel, below the generator-specific parameters and transforms.
        """
        return [
            ChoiceParam(
                name="brush_type",
                label="Brush Type",
                choices=["None", "Stippled", "Multi-Stroke", "Calligraphic"],
                default="None",
                description="Apply a brush effect to the generated paths.",
                randomizable=False,
            ),
            # Stippled brush parameters
            FloatParam(
                name="stipple_spacing_mm",
                label="Stipple Spacing (mm)",
                min=0.1,
                max=20.0,
                step=0.1,
                default=1.0,
                visible_when={"brush_type": ["Stippled"]},
                description="Distance between consecutive dot centres.",
            ),
            FloatParam(
                name="stipple_size_mm",
                label="Stipple Size (mm)",
                min=0.01,
                max=5.0,
                step=0.05,
                default=0.3,
                visible_when={"brush_type": ["Stippled"]},
                description="Dot radius in mm.",
            ),
            FloatParam(
                name="stipple_randomness",
                label="Stipple Randomness",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.2,
                visible_when={"brush_type": ["Stippled"]},
                description="Random variation in dot position and size (0–1).",
            ),
            # Multi-Stroke brush parameters
            IntParam(
                name="stroke_count",
                label="Stroke Count",
                min=1,
                max=10,
                step=1,
                default=3,
                visible_when={"brush_type": ["Multi-Stroke"]},
                description="Number of parallel strokes to draw.",
            ),
            FloatParam(
                name="stroke_spread_mm",
                label="Stroke Spread (mm)",
                min=0.0,
                max=10.0,
                step=0.1,
                default=0.5,
                visible_when={"brush_type": ["Multi-Stroke"]},
                description="Maximum lateral offset in mm.",
            ),
            FloatParam(
                name="stroke_noise",
                label="Stroke Noise",
                min=0.0,
                max=1.0,
                step=0.05,
                default=0.3,
                visible_when={"brush_type": ["Multi-Stroke"]},
                description="Per-point noise intensity (0–1).",
            ),
            # Calligraphic brush parameters
            FloatParam(
                name="nib_angle",
                label="Nib Angle (deg)",
                min=0.0,
                max=180.0,
                step=5.0,
                default=45.0,
                visible_when={"brush_type": ["Calligraphic"]},
                description="Pen nib angle in degrees.",
            ),
            FloatParam(
                name="nib_width_mm",
                label="Nib Width (mm)",
                min=0.05,
                max=10.0,
                step=0.1,
                default=1.5,
                visible_when={"brush_type": ["Calligraphic"]},
                description="Maximum total stroke width.",
            ),
            FloatParam(
                name="min_width_mm",
                label="Min Width (mm)",
                min=0.01,
                max=5.0,
                step=0.05,
                default=0.2,
                visible_when={"brush_type": ["Calligraphic"]},
                description="Minimum total stroke width.",
            ),
        ]
