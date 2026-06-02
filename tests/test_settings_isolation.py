"""Guards that the test suite can never read or write the developer's real
``QSettings`` store.

This exists because of a real incident: dialog tests that construct
``QSettings("Plottter", "Plottter")`` and ``remove()``/``clear()`` keys, run
under ``timeout --signal=KILL``, erased a developer's real settings (the AxiDraw
``bed_size`` among them) when SIGKILL landed before a ``finally`` restore. The
session-level redirect in ``conftest.py`` prevents that; these tests fail loudly
if that redirect is ever weakened or removed.
"""

from __future__ import annotations

import os


def test_qsettings_resolves_outside_real_config():
    """The app's settings object must not point at the real config file."""
    from PyQt6.QtCore import QSettings

    path = QSettings("Plottter", "Plottter").fileName()
    real = os.path.join(os.path.expanduser("~"), ".config", "Plottter")
    assert not os.path.abspath(path).startswith(os.path.abspath(real)), (
        f"QSettings resolved to the real config dir ({path}); the conftest "
        "isolation redirect is not in effect."
    )
    assert "plottter-test-config-" in path, (
        f"QSettings did not resolve into the sandbox config dir: {path}"
    )


def test_destructive_settings_ops_do_not_touch_real_config(tmp_path):
    """clear()/setValue() — what dialog tests do — must stay in the sandbox."""
    from PyQt6.QtCore import QSettings

    real_conf = os.path.join(
        os.path.expanduser("~"), ".config", "Plottter", "Plottter.conf"
    )
    before = os.path.getmtime(real_conf) if os.path.exists(real_conf) else None

    settings = QSettings("Plottter", "Plottter")
    settings.setValue("axidraw/bed_size", "SANDBOX ONLY — must never reach real config")
    settings.clear()
    settings.sync()

    after = os.path.getmtime(real_conf) if os.path.exists(real_conf) else None
    assert before == after, "Real Plottter.conf was modified by a test."
