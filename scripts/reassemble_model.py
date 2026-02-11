#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_parts(parts_dir: Path, manifest: Optional[Dict]) -> Tuple[List[Path], Optional[List[Dict]]]:
    if manifest:
        manifest_parts = manifest.get("parts", [])
        if not manifest_parts:
            raise ValueError("Manifest does not contain parts.")
        parts = [parts_dir / part["file"] for part in manifest_parts]
        return parts, manifest_parts

    pattern = re.compile(r".*\.part\d+$")
    parts = [p for p in parts_dir.iterdir() if p.is_file() and pattern.match(p.name)]
    if not parts:
        raise ValueError("No part files found. Expected files like model.part000.")
    return sorted(parts, key=lambda p: p.name), None


def verify_parts(parts: List[Path], manifest_parts: Optional[List[Dict]]):
    if not manifest_parts:
        return

    if len(parts) != len(manifest_parts):
        raise ValueError("Manifest parts count does not match available files.")

    for part_path, part_info in zip(parts, manifest_parts):
        expected = (part_info.get("sha256") or "").lower()
        if not expected:
            continue
        actual = sha256_file(part_path)
        if actual.lower() != expected:
            raise ValueError(f"Checksum mismatch for {part_path.name}")


def reassemble(parts: List[Path], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_handle:
        for part_path in parts:
            with part_path.open("rb") as part_handle:
                for chunk in iter(lambda: part_handle.read(8 * 1024 * 1024), b""):
                    output_handle.write(chunk)


def verify_output(output_path: Path, manifest: Optional[Dict]):
    if not manifest:
        return

    expected = (manifest.get("original_sha256") or "").lower()
    if not expected:
        return

    actual = sha256_file(output_path)
    if actual.lower() != expected:
        raise ValueError("Reassembled file checksum mismatch.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reassemble model parts into a GGUF file.")
    parser.add_argument("--parts-dir", required=True, help="Directory containing split parts")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--manifest", help="Path to manifest.json")
    parser.add_argument("--verify", action="store_true", help="Verify checksums using manifest")
    args = parser.parse_args()

    parts_dir = Path(args.parts_dir)
    if not parts_dir.exists():
        raise FileNotFoundError(f"Parts directory not found: {parts_dir}")

    manifest = None
    if args.manifest:
        manifest = load_manifest(Path(args.manifest))

    if args.verify and not manifest:
        raise ValueError("--verify requires --manifest.")

    parts, manifest_parts = resolve_parts(parts_dir, manifest)
    missing = [p for p in parts if not p.exists()]
    if missing:
        missing_names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(f"Missing parts: {missing_names}")

    if args.verify:
        verify_parts(parts, manifest_parts)

    output_path = Path(args.output)
    reassemble(parts, output_path)

    if args.verify:
        verify_output(output_path, manifest)

    print(f"Reassembled model written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
