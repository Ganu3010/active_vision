"""
diagnose_dataset.py
====================
Run this to inspect your ShapeNetCore directory layout before using the env.

Usage:
    python diagnose_dataset.py --root data/ShapeNetCore
    python diagnose_dataset.py --root data/ShapeNetCore --depth 5
"""

import argparse
from pathlib import Path


def print_tree(root: Path, max_depth: int = 4, max_per_dir: int = 5):
    def _human(size: int) -> str:
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def _walk(path: Path, depth: int, prefix: str = ""):
        if depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir())
        except PermissionError:
            print(f"{prefix}  [permission denied]")
            return
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        for f in files[:max_per_dir]:
            print(f"{prefix}  FILE  {f.name}  ({_human(f.stat().st_size)})")
        if len(files) > max_per_dir:
            print(f"{prefix}  ... +{len(files)-max_per_dir} more files")
        for i, d in enumerate(dirs[:max_per_dir]):
            print(f"{prefix}  DIR   {d.name}/")
            _walk(d, depth + 1, prefix + "      ")
        if len(dirs) > max_per_dir:
            print(f"{prefix}  ... +{len(dirs)-max_per_dir} more dirs")

    print(f"\n{'='*60}")
    print(f"  Path: {root.resolve()}")
    print(f"  Exists: {root.exists()}")
    print(f"{'='*60}")
    if root.exists():
        _walk(root, 0)


def find_objs(root: Path, limit: int = 10):
    print(f"\n{'='*60}")
    print("  Searching for .obj files ...")
    print(f"{'='*60}")
    found = list(root.rglob("*.obj"))
    if not found:
        print("  ❌  No .obj files found!")
        for ext in ["*.glb", "*.ply", "*.stl", "*.off", "*.dae"]:
            alts = list(root.rglob(ext))[:2]
            if alts:
                print(f"  Found {ext}: {[str(p.relative_to(root)) for p in alts]}")
    else:
        print(f"  Found {len(found)} .obj file(s). First {min(limit,len(found))}:\n")
        for p in found[:limit]:
            print(f"    {p.relative_to(root)}")
        print(f"\n  Depth breakdown of: {found[0].relative_to(root)}")
        for i, part in enumerate(found[0].relative_to(root).parts):
            print(f"    Level {i}: {part}")


def check_zips(root: Path):
    zips = list(root.rglob("*.zip"))[:5]
    if zips:
        print(f"\n⚠️  WARNING: Found zip files — dataset may not be extracted!")
        for z in zips:
            print(f"  {z.relative_to(root)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ShapeNetCore")
    parser.add_argument("--depth", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.root)
    print_tree(root, max_depth=args.depth)
    check_zips(root)
    find_objs(root)
    print(f"\n{'='*60}")
    print("  Paste this output when reporting issues.")
    print(f"{'='*60}\n")