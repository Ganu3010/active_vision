"""
extract_shapenet.py
====================
Extracts all ShapeNetCore zip files in-place.

Usage:
    python extract_shapenet.py --root data/ShapeNetCore
    python extract_shapenet.py --root data/ShapeNetCore --workers 4   # parallel
    python extract_shapenet.py --root data/ShapeNetCore --dry-run     # preview only
"""

import argparse
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def extract_zip(zip_path: Path, dest: Path, dry_run: bool = False):
    """Extract a single zip file. Returns (name, success, message)."""
    name = zip_path.name
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            if dry_run:
                return name, True, f"  [dry-run] would extract {len(members)} files"

            # Check if already extracted (skip if top-level dir exists)
            # ShapeNetCore zips typically contain a folder named after the synset
            top_dirs = {m.split("/")[0] for m in members if "/" in m}
            already_done = all((dest / d).is_dir() for d in top_dirs if d)
            if already_done and top_dirs:
                return name, True, "  already extracted, skipping"

            zf.extractall(dest)
            return name, True, f"  extracted {len(members)} files → {dest}"
    except zipfile.BadZipFile as e:
        return name, False, f"  ❌ bad zip: {e}"
    except Exception as e:
        return name, False, f"  ❌ error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Extract ShapeNetCore zip files")
    parser.add_argument("--root", default="data/ShapeNetCore",
                        help="Path to folder containing the .zip files")
    parser.add_argument("--dest", default=None,
                        help="Destination folder (default: same as --root)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel extraction threads (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be extracted without doing it")
    args = parser.parse_args()

    root = Path(args.root)
    dest = Path(args.dest) if args.dest else root

    if not root.exists():
        print(f"❌  Root path does not exist: {root.resolve()}")
        return

    zips = sorted(root.glob("*.zip"))
    if not zips:
        print(f"No .zip files found in {root.resolve()}")
        # Check if already extracted
        obj_count = sum(1 for _ in root.rglob("*.obj"))
        if obj_count:
            print(f"✅  Found {obj_count} .obj files — dataset appears already extracted!")
        return

    total_size = sum(z.stat().st_size for z in zips)
    print(f"\nFound {len(zips)} zip file(s)  "
          f"(total compressed: {total_size / 1e9:.1f} GB)\n")
    if args.dry_run:
        print("DRY RUN — nothing will be extracted.\n")

    if args.workers > 1 and not args.dry_run:
        # Parallel extraction
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(extract_zip, z, dest, args.dry_run): z for z in zips}
            for i, fut in enumerate(as_completed(futures), 1):
                name, ok, msg = fut.result()
                status = "✅" if ok else "❌"
                print(f"[{i:>3}/{len(zips)}] {status}  {name}\n{msg}")
    else:
        # Sequential (safer, less memory pressure)
        for i, z in enumerate(zips, 1):
            print(f"[{i:>3}/{len(zips)}]  {z.name}  "
                  f"({z.stat().st_size / 1e6:.0f} MB)", end="", flush=True)
            name, ok, msg = extract_zip(z, dest, args.dry_run)
            status = " ✅" if ok else " ❌"
            print(f"{status}\n{msg}")

    print("\nDone.")
    if not args.dry_run:
        obj_count = sum(1 for _ in dest.rglob("*.obj"))
        print(f"✅  Total .obj files now available: {obj_count:,}")


if __name__ == "__main__":
    main()