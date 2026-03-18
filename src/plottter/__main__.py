"""Entry point: routes to CLI or GUI based on arguments."""

import sys


# Flags that unambiguously indicate CLI/headless mode.
_CLI_FLAGS = frozenset({
    "--help", "-h",
    "--list-generators",
    "--list-presets",
    "--generator", "-g",
    "--output", "-o",
    "--preset", "-p",
    "--format", "-f",
})


def main() -> None:
    args = sys.argv[1:]
    is_cli = any(arg in _CLI_FLAGS for arg in args)

    if is_cli:
        from plottter.cli import run_cli
        sys.exit(run_cli(args))
    else:
        from plottter import app
        app.main()


if __name__ == "__main__":
    main()
