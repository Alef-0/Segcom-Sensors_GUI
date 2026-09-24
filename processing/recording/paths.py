"""Safe path helpers for current and legacy recording layouts."""

from pathlib import Path

IMAGE_DIRECTORY_NAME = "images"
POINT_CLOUD_DIRECTORY_NAME = "point_cloud"


def _reference(directory: str, filename: str | Path) -> str:
    return (Path(directory) / Path(str(filename)).name).as_posix()


def image_reference(filename: str | Path) -> str:
    return _reference(IMAGE_DIRECTORY_NAME, filename)


def point_cloud_reference(filename: str | Path) -> str:
    return _reference(POINT_CLOUD_DIRECTORY_NAME, filename)


def image_path(folder: Path, filename: str | Path) -> Path:
    return folder / image_reference(filename)


def point_cloud_path(folder: Path, filename: str | Path) -> Path:
    return folder / point_cloud_reference(filename)


def resolve_recording_file(
    folder: Path,
    reference: str | Path | None,
    directory_name: str,
) -> Path | None:
    """Resolve a contained file reference, including bare legacy filenames."""
    if not reference:
        return None
    root = folder.expanduser().resolve()
    raw = Path(str(reference))
    candidates = [root / raw]
    if len(raw.parts) == 1:
        candidates.append(root / directory_name / raw.name)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(root) and resolved.is_file():
            return resolved
    return None
