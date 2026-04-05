"""ProcessingPlugin ABC and registry for post-processing plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plottter.generators.base import Parameter
    from plottter.models.path import Polyline

#: Registry of all registered processing plugins, keyed by plugin name.
PROCESSING_PLUGINS: dict[str, type[ProcessingPlugin]] = {}


def register_processing_plugin(cls: type[ProcessingPlugin]) -> type[ProcessingPlugin]:
    """Decorator to register a :class:`ProcessingPlugin` subclass by its name."""
    PROCESSING_PLUGINS[cls.name] = cls
    return cls


class ProcessingPlugin(ABC):
    """Abstract base class for post-processing plugins.

    A processing plugin transforms a layer's polylines (e.g. smoothing,
    jitter, scaling) and is invoked from the Tools menu on the active layer.

    Subclass this and decorate with :func:`register_processing_plugin` to
    register a processing plugin. Place the file in any of the plugin
    directories scanned by :mod:`plottter.generators.plugin_loader`.

    Example::

        from plottter.processing.plugin import ProcessingPlugin, register_processing_plugin
        from plottter.generators.base import FloatParam

        @register_processing_plugin
        class JitterPlugin(ProcessingPlugin):
            name = "Jitter Paths"
            description = "Randomly displace path points by a small amount."

            def get_parameters(self):
                return [
                    FloatParam("amount_mm", "Jitter Amount (mm)",
                               min=0.0, max=10.0, step=0.1, default=1.0),
                ]

            def process(self, paths, params):
                import random
                amount = params.get("amount_mm", 1.0)
                result = []
                for path in paths:
                    new_path = [
                        (x + random.uniform(-amount, amount),
                         y + random.uniform(-amount, amount))
                        for x, y in path
                    ]
                    result.append(new_path)
                return result
    """

    #: Display name shown in the Tools menu (must be unique).
    name: str = ""

    #: Short description shown as a tooltip.
    description: str = ""

    def get_parameters(self) -> list[Parameter]:
        """Return the list of parameters for this plugin.

        Return an empty list if the plugin requires no parameters.
        The parameters are rendered as simple input controls in a dialog
        before the plugin runs.
        """
        return []

    @abstractmethod
    def process(
        self,
        paths: list[Polyline],
        params: dict,
    ) -> list[Polyline]:
        """Transform *paths* using *params* and return the modified paths.

        Parameters
        ----------
        paths:
            The active layer's polylines (each polyline is a list of
            ``(x_mm, y_mm)`` tuples).  Do not mutate in-place — return a
            new list.
        params:
            Dict of parameter values keyed by parameter name, matching the
            names returned by :meth:`get_parameters`.

        Returns
        -------
        list[Polyline]
            The transformed polylines.
        """
