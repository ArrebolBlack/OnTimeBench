from pathlib import Path

_PACKAGE = Path(__file__).resolve().parent


def resource_path(*parts: str) -> Path:
    """Locate benchmark resources in a source checkout or an installed wheel."""
    if _PACKAGE.parent.name == "src" and (_PACKAGE.parents[1] / "pyproject.toml").is_file():
        return _PACKAGE.parents[1].joinpath(*parts)
    return _PACKAGE.joinpath(*parts)
