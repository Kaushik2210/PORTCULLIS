"""Scaffold: the package resolves under the shared `portcullis` namespace."""

import importlib


def test_package_imports() -> None:
    mod = importlib.import_module("portcullis.gateway")
    assert mod.__name__ == "portcullis.gateway"


def test_package_is_typed() -> None:
    """py.typed must ship, or downstream mypy --strict silently degrades to Any."""
    import pathlib

    mod = importlib.import_module("portcullis.gateway")
    assert mod.__file__ is not None
    assert (pathlib.Path(mod.__file__).parent / "py.typed").is_file()
