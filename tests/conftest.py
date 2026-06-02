"""Shared pytest fixtures and hooks.

Settings isolation
------------------
The app persists state via ``QSettings("Plottter", "Plottter")`` — on Linux
that is ``~/.config/Plottter/Plottter.conf``. Several test modules construct
dialogs (AxiDraw, New Project, …) that read *and write* those same keys, and
some deliberately ``remove()``/``clear()`` them. Because the suite is run under
``timeout --signal=KILL`` (QApplication teardown hangs headless — see
CLAUDE.md), a SIGKILL can land before a test's ``finally`` restore runs, leaving
the developer's real settings mutated or erased.

To make that impossible, we redirect the entire config location to a throwaway
directory **before Qt is imported**. ``XDG_CONFIG_HOME`` is read dynamically by
Qt's ``QStandardPaths`` on Unix, so setting it here (at conftest import, before
any ``QApplication``/``QSettings`` exists) sends every NativeFormat UserScope
QSettings into the temp dir. The autouse session fixture below adds
belt-and-suspenders ``setPath`` redirects and restores the global default
format, so even a test that calls ``setDefaultFormat``/``setPath`` itself cannot
leak outside the sandbox or reach the real config.
"""

from __future__ import annotations

import os
import tempfile

# Must happen at import time — before any test (or pytest plugin) imports PyQt6
# and resolves a settings path. Creating the dir up front and pointing the
# config root at it guarantees QSettings never sees the real ~/.config.
_TEST_CONFIG_DIR = tempfile.mkdtemp(prefix="plottter-test-config-")
os.environ["XDG_CONFIG_HOME"] = _TEST_CONFIG_DIR
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolate_qsettings():
    """Pin every QSettings format/scope to the throwaway config dir.

    ``XDG_CONFIG_HOME`` (set at import above) already covers the common case
    (NativeFormat / UserScope on Linux). This additionally redirects the
    explicit IniFormat path some tests select via ``setDefaultFormat`` and
    restores the process-global default format afterwards, so no test can leave
    that global flipped for later tests — or for anything sharing the process.
    """
    from PyQt6.QtCore import QSettings

    prev_default = QSettings.defaultFormat()
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, _TEST_CONFIG_DIR
    )
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.SystemScope, _TEST_CONFIG_DIR
    )
    try:
        yield
    finally:
        QSettings.setDefaultFormat(prev_default)
