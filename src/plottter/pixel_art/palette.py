"""Base palette classes and utilities.

This module provides the foundational palette system for the pixel art converter.
Palettes are collections of RGB colors that define the available colors for
pixel art conversion.

Key classes:
    Palette: Abstract base class for all palettes
    FixedPalette: Palette with a predefined set of colors
    GeneratedPalette: Palette generated from color bit depth
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from plottter.pixel_art.exceptions import InvalidPaletteError

# Type alias for RGB color tuple
RGB = Tuple[int, int, int]


@dataclass
class PaletteMetadata:
    """Metadata about a palette.

    Attributes:
        name: Human-readable palette name
        system: Game system or platform (e.g., "NES", "Game Boy", "Custom")
        year: Year the system was released (optional)
        color_count: Number of colors in the palette
        description: Brief description of the palette
        author: Creator of the palette (for custom palettes)
        source: Source URL or reference
    """

    name: str
    system: str
    year: Optional[int] = None
    color_count: int = 0
    description: str = ""
    author: Optional[str] = None
    source: Optional[str] = None


class Palette(ABC):
    """Abstract base class for all palettes.

    A palette represents a fixed set of colors that can be used for
    pixel art conversion. All palettes must implement the colors
    and metadata properties.

    Subclasses should override the colors and metadata properties
    to provide their specific implementation.
    """

    @property
    @abstractmethod
    def colors(self) -> List[RGB]:
        """Return list of RGB color tuples.

        Returns:
            List of (R, G, B) tuples where each value is 0-255
        """
        ...

    @property
    @abstractmethod
    def metadata(self) -> PaletteMetadata:
        """Return palette metadata.

        Returns:
            PaletteMetadata instance with palette information
        """
        ...

    @property
    def color_count(self) -> int:
        """Return number of colors in the palette."""
        return len(self.colors)

    @property
    def name(self) -> str:
        """Return palette name."""
        return self.metadata.name

    def get_color(self, index: int) -> RGB:
        """Get color by index.

        Args:
            index: Color index (0-based)

        Returns:
            RGB tuple

        Raises:
            InvalidPaletteError: If index is out of range
        """
        colors = self.colors
        if index < 0 or index >= len(colors):
            raise InvalidPaletteError(f"Color index {index} out of range [0, {len(colors) - 1}]")
        return colors[index]

    def find_color_index(self, color: RGB) -> Optional[int]:
        """Find the index of a color in the palette.

        Args:
            color: RGB tuple to find

        Returns:
            Index of the color, or None if not found
        """
        try:
            return self.colors.index(color)
        except ValueError:
            return None

    def contains(self, color: RGB) -> bool:
        """Check if a color is in the palette.

        Args:
            color: RGB tuple to check

        Returns:
            True if color is in palette
        """
        return color in self.colors

    def __iter__(self) -> Iterator[RGB]:
        """Iterate over colors in the palette."""
        return iter(self.colors)

    def __len__(self) -> int:
        """Return number of colors."""
        return self.color_count

    def __contains__(self, color: RGB) -> bool:
        """Check if color is in palette."""
        return self.contains(color)

    def __getitem__(self, index: int) -> RGB:
        """Get color by index."""
        return self.get_color(index)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize palette to dictionary.

        Returns:
            Dictionary representation of the palette
        """
        return {
            "metadata": {
                "name": self.metadata.name,
                "system": self.metadata.system,
                "year": self.metadata.year,
                "color_count": self.color_count,
                "description": self.metadata.description,
                "author": self.metadata.author,
                "source": self.metadata.source,
            },
            "colors": [list(c) for c in self.colors],
        }

    def to_hex_list(self) -> List[str]:
        """Convert colors to hex string list.

        Returns:
            List of hex color strings (e.g., ["#FF0000", "#00FF00"])
        """
        return [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in self.colors]


class FixedPalette(Palette):
    """Palette with a fixed set of colors.

    This is the most common palette type, used for predefined
    retro game palettes and custom user palettes.

    Example:
        >>> colors = [(0, 0, 0), (255, 255, 255)]
        >>> metadata = PaletteMetadata(name="B&W", system="Custom")
        >>> palette = FixedPalette(colors, metadata)
    """

    def __init__(self, colors: List[RGB], metadata: PaletteMetadata):
        """Initialize fixed palette.

        Args:
            colors: List of RGB color tuples
            metadata: Palette metadata

        Raises:
            InvalidPaletteError: If colors list is empty or invalid
        """
        if not colors:
            raise InvalidPaletteError("Palette cannot be empty")

        # Validate and store colors
        self._colors: List[RGB] = []
        for i, color in enumerate(colors):
            if not isinstance(color, (tuple, list)) or len(color) != 3:
                raise InvalidPaletteError(
                    f"Color at index {i} must be an RGB tuple, got {type(color)}"
                )
            r, g, b = color
            if not all(isinstance(c, int) and 0 <= c <= 255 for c in (r, g, b)):
                raise InvalidPaletteError(f"Color at index {i} has invalid RGB values: {color}")
            self._colors.append((int(r), int(g), int(b)))

        self._metadata = metadata
        self._metadata.color_count = len(self._colors)

    @property
    def colors(self) -> List[RGB]:
        """Return copy of colors list."""
        return self._colors.copy()

    @property
    def metadata(self) -> PaletteMetadata:
        """Return palette metadata."""
        return self._metadata

    @classmethod
    def from_hex_list(
        cls,
        hex_colors: List[str],
        name: str,
        system: str = "Custom",
        **metadata_kwargs: Any,
    ) -> "FixedPalette":
        """Create palette from hex color strings.

        Args:
            hex_colors: List of hex strings (e.g., ["#FF0000", "00FF00"])
            name: Palette name
            system: System/platform name
            **metadata_kwargs: Additional metadata fields

        Returns:
            FixedPalette instance

        Raises:
            InvalidPaletteError: If hex colors are invalid
        """
        colors: List[RGB] = []
        for i, hex_color in enumerate(hex_colors):
            # Remove # prefix if present
            hex_str = hex_color.lstrip("#")

            if len(hex_str) != 6:
                raise InvalidPaletteError(f"Invalid hex color at index {i}: {hex_color}")

            try:
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
                colors.append((r, g, b))
            except ValueError as e:
                raise InvalidPaletteError(f"Invalid hex color at index {i}: {hex_color}") from e

        metadata = PaletteMetadata(name=name, system=system, **metadata_kwargs)
        return cls(colors, metadata)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FixedPalette":
        """Create palette from dictionary.

        Args:
            data: Dictionary with 'colors' and 'metadata' keys

        Returns:
            FixedPalette instance

        Raises:
            InvalidPaletteError: If data is invalid
        """
        if "colors" not in data:
            raise InvalidPaletteError("Palette data must contain 'colors' key")

        colors = [tuple(c) for c in data["colors"]]

        meta_data = data.get("metadata", {})
        metadata = PaletteMetadata(
            name=meta_data.get("name", "Unnamed"),
            system=meta_data.get("system", "Custom"),
            year=meta_data.get("year"),
            description=meta_data.get("description", ""),
            author=meta_data.get("author"),
            source=meta_data.get("source"),
        )

        return cls(colors, metadata)


class GeneratedPalette(Palette):
    """Palette generated from color bit depth.

    This palette type generates all possible colors for a given
    bit depth per channel. For example, 2 bits per channel gives
    4 levels per channel (0, 85, 170, 255) = 64 total colors.

    Useful for palettes like SNES (15-bit, 5 bits per channel)
    or Sega Genesis (9-bit, 3 bits per channel).

    Example:
        >>> metadata = PaletteMetadata(name="EGA", system="IBM PC")
        >>> palette = GeneratedPalette(2, metadata)  # 2 bits = 64 colors
    """

    def __init__(self, bits_per_channel: int, metadata: PaletteMetadata):
        """Initialize generated palette.

        Args:
            bits_per_channel: Number of bits per color channel (1-8)
            metadata: Palette metadata

        Raises:
            InvalidPaletteError: If bits_per_channel is invalid
        """
        if not isinstance(bits_per_channel, int) or bits_per_channel < 1:
            raise InvalidPaletteError(
                f"bits_per_channel must be a positive integer, got {bits_per_channel}"
            )
        if bits_per_channel > 8:
            raise InvalidPaletteError(f"bits_per_channel cannot exceed 8, got {bits_per_channel}")

        self._bits = bits_per_channel
        self._metadata = metadata
        self._colors = self._generate_colors()
        self._metadata.color_count = len(self._colors)

    def _generate_colors(self) -> List[RGB]:
        """Generate all colors for the bit depth.

        Returns:
            List of RGB tuples
        """
        # Number of levels per channel
        levels = 2**self._bits

        # Generate evenly spaced values 0-255
        if levels == 1:
            values = [0]
        else:
            values = [int(round(i * 255 / (levels - 1))) for i in range(levels)]

        # Generate all combinations
        colors: List[RGB] = []
        for r in values:
            for g in values:
                for b in values:
                    colors.append((r, g, b))

        return colors

    @property
    def colors(self) -> List[RGB]:
        """Return copy of generated colors."""
        return self._colors.copy()

    @property
    def metadata(self) -> PaletteMetadata:
        """Return palette metadata."""
        return self._metadata

    @property
    def bits_per_channel(self) -> int:
        """Return bits per channel."""
        return self._bits

    @property
    def levels_per_channel(self) -> int:
        """Return number of levels per channel."""
        return int(2**self._bits)


class SubPalette(Palette):
    """A subset of colors from a parent palette.

    Useful for working with palette pages or subsets of larger
    palettes, such as NES's 4-color sprite palettes.

    Example:
        >>> parent = FixedPalette([...], metadata)
        >>> sub = SubPalette(parent, indices=[0, 1, 4, 8], name="Sprite 0")
    """

    def __init__(
        self,
        parent: Palette,
        indices: List[int],
        name: Optional[str] = None,
    ):
        """Initialize sub-palette.

        Args:
            parent: Parent palette to extract colors from
            indices: List of color indices to include
            name: Optional name (defaults to parent name + " (subset)")

        Raises:
            InvalidPaletteError: If indices are invalid
        """
        if not indices:
            raise InvalidPaletteError("SubPalette indices cannot be empty")

        parent_colors = parent.colors
        max_index = len(parent_colors) - 1

        for i, idx in enumerate(indices):
            if idx < 0 or idx > max_index:
                raise InvalidPaletteError(
                    f"Index {idx} at position {i} is out of range [0, {max_index}]"
                )

        self._parent = parent
        self._indices = list(indices)
        self._colors = [parent_colors[i] for i in indices]

        # Create metadata for sub-palette
        parent_meta = parent.metadata
        self._metadata = PaletteMetadata(
            name=name or f"{parent_meta.name} (subset)",
            system=parent_meta.system,
            year=parent_meta.year,
            color_count=len(self._colors),
            description=f"Subset of {parent_meta.name}",
        )

    @property
    def colors(self) -> List[RGB]:
        """Return colors in the sub-palette."""
        return self._colors.copy()

    @property
    def metadata(self) -> PaletteMetadata:
        """Return sub-palette metadata."""
        return self._metadata

    @property
    def parent(self) -> Palette:
        """Return parent palette."""
        return self._parent

    @property
    def indices(self) -> List[int]:
        """Return indices into parent palette."""
        return self._indices.copy()

    def get_parent_index(self, local_index: int) -> int:
        """Convert local index to parent palette index.

        Args:
            local_index: Index in this sub-palette

        Returns:
            Corresponding index in parent palette

        Raises:
            InvalidPaletteError: If index is out of range
        """
        if local_index < 0 or local_index >= len(self._indices):
            raise InvalidPaletteError(
                f"Local index {local_index} out of range [0, {len(self._indices) - 1}]"
            )
        return self._indices[local_index]
