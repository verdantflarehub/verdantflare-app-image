"""Shared validation for project-relative storage paths."""
from pathlib import Path


def validate_project_id(project_id: str) -> None:
    if (not isinstance(project_id, str) or not project_id
            or "\\" in project_id
            or any(part in ("", ".", "..") for part in project_id.split("/"))
            or any(ord(c) < 32 for c in project_id)):
        raise ValueError("Invalid project_id")


def project_path(root: Path, project_id: str) -> Path:
    validate_project_id(project_id)
    path = root / project_id / "artifacts"
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Project path escapes storage root")
    return path
