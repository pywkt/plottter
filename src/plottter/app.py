"""Application entry point — creates QApplication and launches MainWindow."""

from __future__ import annotations

import sys


def main() -> None:
    from PyQt6.QtWidgets import QApplication
    from plottter.gui.main_window import MainWindow
    from plottter.gui.project_controller import ProjectController
    from plottter.models import Canvas, Layer, Project

    app = QApplication(sys.argv)
    app.setApplicationName("Plottter")
    app.setApplicationVersion("0.1.0")

    # Default project: A4 canvas with one empty "Layer 1"
    canvas = Canvas.from_preset("A4", margin=10.0)
    project = Project(name="Untitled", canvas=canvas)
    project.add_layer(Layer(name="Layer 1", color="#000000"))

    controller = ProjectController(project)
    window = MainWindow(controller)
    window.show()

    sys.exit(app.exec())
