"""Side-by-side MuJoCo comparison of two trained hero policies."""

__version__ = "0.1.0"


def main() -> int:
    """Console-script entry point."""
    from .app import main as _main

    return _main()


__all__ = ["main", "__version__"]