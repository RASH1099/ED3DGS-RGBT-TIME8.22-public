#!/usr/bin/env python3
"""Static release audit for the source-only E-D3DGS-RGBT package."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TOP = {
    ".gitattributes", ".gitignore", "LICENSE.md", "MANIFEST.sha256",
    "README.md", "RELEASE_COMPLETE", "environment.txt", "requirements.txt",
    "run.sh", "train.py", "render.py", "metrics.py", "arguments",
    "gaussian_renderer", "outputs", "scene", "submodules", "time_alignment",
    "utils",
}
PUBLIC_SCRIPTS = {"run.sh", "train.py", "render.py", "metrics.py"}
EXPECTED_CONFIGS = {
    "arguments/__init__.py", "arguments/default.py", "arguments/covers.py",
    "arguments/covers_rgb_teacher.py", "arguments/MeetingRoom.py",
    "arguments/MeetingRoom_rgb_teacher.py", "arguments/PourHotWater.py",
    "arguments/PourHotWater_rgb_teacher.py",
    "arguments/Heatingtable.py", "arguments/Heatingtable_rgb_teacher.py",
    "arguments/HotPressMachine.py", "arguments/HotPressMachine_rgb_teacher.py",
    "arguments/Hotwind.py", "arguments/Hotwind_rgb_teacher.py",
    "arguments/Bacon.py", "arguments/Bacon_rgb_teacher.py",
    "arguments/DeliverIcePacks.py", "arguments/DeliverIcePacks_rgb_teacher.py",
    "arguments/HairDryer.py", "arguments/HairDryer_rgb_teacher.py",
    "arguments/HairDryerDark.py", "arguments/HairDryerDark_rgb_teacher.py",
    "arguments/IroningClothes.py", "arguments/IroningClothes_rgb_teacher.py",
    "arguments/LightTheCandles.py", "arguments/LightTheCandles_rgb_teacher.py",
    "arguments/WhiteFoamCovers.py", "arguments/WhiteFoamCovers_rgb_teacher.py",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_manifest(path: Path) -> int:
    require(path.is_file(), f"Missing manifest: {path}")
    count = 0
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, f"Malformed manifest line {line_number}")
        digest, relative = fields
        relative = relative.lstrip("* ")
        target = Path(relative)
        require(not target.is_absolute(), f"Absolute manifest path: {target}")
        resolved = (ROOT / target).resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError as exc:
            raise RuntimeError(f"Manifest escapes release: {target}") from exc
        require(resolved.is_file(), f"Missing manifest file: {target}")
        hasher = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        require(hasher.hexdigest() == digest, f"Hash mismatch: {target}")
        count += 1
    require(count > 0, "Empty manifest")
    return count


def verify_layout() -> tuple[int, int]:
    actual_top = {path.name for path in ROOT.iterdir()}
    require(REQUIRED_TOP <= actual_top,
            f"Missing top-level entries: {sorted(REQUIRED_TOP - actual_top)}")
    unexpected = actual_top - (REQUIRED_TOP | {".git"})
    require(not unexpected, f"Unexpected top-level entries: {sorted(unexpected)}")
    public = {path.name for path in ROOT.iterdir()
              if path.is_file() and path.suffix in {".py", ".sh"}}
    require(public == PUBLIC_SCRIPTS, f"Unexpected public scripts: {sorted(public)}")

    scripts = 0
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if relative.parts and relative.parts[0] == ".git":
            continue
        require(not path.is_symlink(), f"Symlink is not portable: {relative}")
        require("__pycache__" not in relative.parts,
                f"Python cache in release: {relative}")
        require(not path.name.endswith((".pyc", ".pyo", ".pending", "~")),
                f"Temporary file in release: {relative}")
        require(".egg-info" not in path.parts,
                f"Build metadata in release: {relative}")
        if path.is_file() and path.suffix in {".py", ".sh"}:
            scripts += 1

    require((ROOT / "outputs" / ".gitkeep").is_file(),
            "outputs/.gitkeep is required")
    output_files = [p for p in (ROOT / "outputs").rglob("*")
                    if p.name != ".gitkeep"]
    require(not output_files,
            f"Generated artifacts in source release: {output_files}")
    for relative in EXPECTED_CONFIGS:
        require((ROOT / relative).is_file(),
                f"Missing expected configuration: {relative}")

    run_text = (ROOT / "run.sh").read_text()
    for forbidden in ("dazhi-Super-Server", "/home/jiangzhenghan/",
                      "run_ablation.sh", "run_gate_matrix.sh", "OUTPUTS.sha256"):
        require(forbidden not in run_text,
                f"Machine/experiment-specific launcher text: {forbidden}")
    require(subprocess.run(["bash", "-n", str(ROOT / "run.sh")],
                           check=False).returncode == 0,
            "run.sh failed bash -n")
    return scripts, len(list(ROOT.rglob("*.py")))


def verify_python_sources() -> int:
    count = 0
    for path in ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
        count += 1
    return count


def main() -> None:
    scripts, _ = verify_layout()
    compiled = verify_python_sources()
    code_files = verify_manifest(ROOT / "MANIFEST.sha256")
    metadata = json.loads((ROOT / "RELEASE_COMPLETE").read_text())
    require(metadata.get("status") == "PASS", "Release marker is not PASS")
    require(metadata.get("protocol") == "strict_v2_source_release",
            "Release marker protocol mismatch")
    print(json.dumps({
        "status": "PASS",
        "repository": "E-D3DGS-RGBT",
        "protocol": "strict_v2_source_release",
        "public_scripts": sorted(PUBLIC_SCRIPTS),
        "python_shell_file_count": scripts,
        "python_sources_compiled": compiled,
        "manifest_files_verified": code_files,
        "official_scene_families": ["Lab", "MeetingRoom"],
        "portable_outputs": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
