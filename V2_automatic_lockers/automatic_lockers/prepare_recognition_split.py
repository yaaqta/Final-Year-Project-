"""
prepare_recognition_split.py

Generates `recognition_split.json` describing how each test video maps to:
  - identity       : person name parsed from filename
  - group          : "normal" | "mask" | "spoof"
  - role           : "registered" | "stranger"
  - split          : "dev" | "test"
  - ground_truth   : identity (for registered) or "Unknown" (for stranger)

Filename pattern: <class>_<name>_<suffix>.mp4
  - user_<name>_normal.mp4   -> group=normal
  - mask_<name>.mp4          -> group=mask
  - spoof_<name>_phone.mp4   -> group=spoof

Modes:
  --auto         pick registered/stranger and dev/test randomly with seed
  --interactive  prompt for each identity
  --from_config  read role+split from a plain-text config

The script does NOT copy files; it only creates a JSON manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ----------------------------------------------------------------------
# Filename parsing
# ----------------------------------------------------------------------

GROUP_PATTERNS = [
    (re.compile(r"^user_(.+?)_normal\.mp4$", re.IGNORECASE), "normal"),
    (re.compile(r"^mask_(.+?)\.mp4$", re.IGNORECASE), "mask"),
    (re.compile(r"^spoof_(.+?)_phone\.mp4$", re.IGNORECASE), "spoof"),
    (re.compile(r"^spoof_(.+?)_print\.mp4$", re.IGNORECASE), "spoof")
]


def parse_filename(fname: str) -> Tuple[str, str] | None:
    """Returns (identity, group) or None if pattern not recognized."""
    for pat, group in GROUP_PATTERNS:
        m = pat.match(fname)
        if m:
            return m.group(1), group
    return None


def scan_videos(videos_dir: Path) -> Dict[str, Dict]:
    """Returns {filename: {identity, group}} for every .mp4 we can parse."""
    out: Dict[str, Dict] = {}
    if not videos_dir.is_dir():
        print(f"[ERROR] videos_dir not found: {videos_dir}", file=sys.stderr)
        sys.exit(1)
    for f in sorted(videos_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() != ".mp4":
            continue
        parsed = parse_filename(f.name)
        if parsed is None:
            print(f"[WARN] Skipped (unknown pattern): {f.name}", file=sys.stderr)
            continue
        identity, group = parsed
        out[f.name] = {"identity": identity, "group": group}
    return out


def group_by_identity(videos: Dict[str, Dict]) -> Dict[str, List[Tuple[str, str]]]:
    """Returns {identity: [(filename, group), ...]}."""
    by_id: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for fname, meta in videos.items():
        by_id[meta["identity"]].append((fname, meta["group"]))
    return dict(sorted(by_id.items(), key=lambda kv: kv[0].lower()))


# ----------------------------------------------------------------------
# Splitting strategies
# ----------------------------------------------------------------------


def auto_split(
    identities: List[str],
    n_registered: int,
    n_stranger: int,
    seed: int,
) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    """Returns (registered, stranger, identity_split) where identity_split
    maps each identity -> 'dev' | 'test'."""
    if n_registered + n_stranger > len(identities):
        print(
            f"[ERROR] Need {n_registered + n_stranger} identities but only "
            f"found {len(identities)}: {identities}",
            file=sys.stderr,
        )
        sys.exit(1)

    rng = random.Random(seed)
    shuffled = identities.copy()
    rng.shuffle(shuffled)

    registered = set(shuffled[:n_registered])
    stranger = set(shuffled[n_registered : n_registered + n_stranger])

    # Split each role half-and-half between dev / test
    identity_split: Dict[str, str] = {}
    for role_set in (registered, stranger):
        role_list = sorted(role_set)
        rng.shuffle(role_list)
        half = len(role_list) // 2
        for name in role_list[:half]:
            identity_split[name] = "dev"
        for name in role_list[half:]:
            identity_split[name] = "test"

    return registered, stranger, identity_split


def interactive_split(
    identities: List[str],
) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    print("\n=== Interactive split ===")
    print("For each identity, enter: <role> <split>")
    print("  role  : R (registered) | S (stranger) | X (skip)")
    print("  split : dev | test")
    print("Example:   R dev    |    S test    |    X\n")

    registered: Set[str] = set()
    stranger: Set[str] = set()
    identity_split: Dict[str, str] = {}

    for name in identities:
        while True:
            raw = input(f"  {name:<12} > ").strip()
            if not raw:
                continue
            parts = raw.split()
            role = parts[0].upper()
            if role == "X":
                break
            if role not in ("R", "S"):
                print("    role must be R / S / X")
                continue
            if len(parts) < 2 or parts[1].lower() not in ("dev", "test"):
                print("    split must be dev or test")
                continue
            split = parts[1].lower()
            (registered if role == "R" else stranger).add(name)
            identity_split[name] = split
            break

    return registered, stranger, identity_split


def from_config_split(
    cfg_path: Path,
    identities: List[str],
) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    if not cfg_path.is_file():
        print(f"[ERROR] Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    valid_ids = {name.lower(): name for name in identities}

    registered: Set[str] = set()
    stranger: Set[str] = set()
    identity_split: Dict[str, str] = {}

    for ln, raw in enumerate(cfg_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            print(f"[ERROR] {cfg_path}:{ln} expects '<name> <R|S> <dev|test>'", file=sys.stderr)
            sys.exit(1)
        name_in, role_in, split_in = parts[0], parts[1].upper(), parts[2].lower()
        if name_in.lower() not in valid_ids:
            print(f"[ERROR] {cfg_path}:{ln} unknown identity '{name_in}'", file=sys.stderr)
            sys.exit(1)
        name = valid_ids[name_in.lower()]
        if role_in not in ("R", "S"):
            print(f"[ERROR] {cfg_path}:{ln} role must be R or S", file=sys.stderr)
            sys.exit(1)
        if split_in not in ("dev", "test"):
            print(f"[ERROR] {cfg_path}:{ln} split must be dev or test", file=sys.stderr)
            sys.exit(1)
        (registered if role_in == "R" else stranger).add(name)
        identity_split[name] = split_in

    return registered, stranger, identity_split


# ----------------------------------------------------------------------
# Manifest builder
# ----------------------------------------------------------------------


def build_manifest(
    videos: Dict[str, Dict],
    registered: Set[str],
    stranger: Set[str],
    identity_split: Dict[str, str],
    mode: str,
    seed: int | None,
) -> Dict:
    video_entries: Dict[str, Dict] = {}
    for fname, meta in sorted(videos.items()):
        identity = meta["identity"]
        if identity in registered:
            role = "registered"
            ground_truth = identity
        elif identity in stranger:
            role = "stranger"
            ground_truth = "Unknown"
        else:
            continue  # identity not assigned -> skip from recognition test
        split = identity_split.get(identity, "test")
        video_entries[fname] = {
            "identity": identity,
            "group": meta["group"],
            "role": role,
            "split": split,
            "ground_truth": ground_truth,
        }

    return {
        "registered_identities": sorted(registered),
        "stranger_identities": sorted(stranger),
        "identity_split": dict(sorted(identity_split.items())),
        "videos": video_entries,
        "mode": mode,
        "seed": seed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def print_summary(manifest: Dict, videos: Dict[str, Dict]) -> None:
    reg = manifest["registered_identities"]
    stg = manifest["stranger_identities"]
    isplit = manifest["identity_split"]
    print("\n=== Recognition split summary ===")
    print(f"Identities discovered : {len(set(m['identity'] for m in videos.values()))}")
    print(f"Registered (in DB)    : {len(reg)} -> {reg}")
    print(f"Stranger (not in DB)  : {len(stg)} -> {stg}")
    dev_reg = [n for n in reg if isplit.get(n) == "dev"]
    test_reg = [n for n in reg if isplit.get(n) == "test"]
    dev_stg = [n for n in stg if isplit.get(n) == "dev"]
    test_stg = [n for n in stg if isplit.get(n) == "test"]
    print(f"Dev  : {len(dev_reg)}R + {len(dev_stg)}S  -> R={dev_reg}, S={dev_stg}")
    print(f"Test : {len(test_reg)}R + {len(test_stg)}S -> R={test_reg}, S={test_stg}")
    print(f"Videos assigned       : {len(manifest['videos'])}")
    unassigned = [
        m["identity"] for m in videos.values() if m["identity"] not in set(reg) | set(stg)
    ]
    if unassigned:
        print(f"Videos skipped (unassigned identities): {sorted(set(unassigned))}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos_dir", default="test_videos", help="Folder containing the .mp4 videos")
    p.add_argument("--output", default="recognition_split.json", help="Output manifest path")
    p.add_argument("--auto", action="store_true", help="Auto mode (random with seed)")
    p.add_argument("--interactive", action="store_true", help="Interactive mode")
    p.add_argument("--from_config", help="Read role+split from a config file")
    p.add_argument("--registered", type=int, default=8, help="Number of registered identities (auto mode)")
    p.add_argument("--stranger", type=int, default=4, help="Number of stranger identities (auto mode)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (auto mode)")
    args = p.parse_args()

    mode_flags = [args.auto, args.interactive, bool(args.from_config)]
    if sum(mode_flags) != 1:
        print("[ERROR] Choose exactly one mode: --auto | --interactive | --from_config", file=sys.stderr)
        sys.exit(1)

    videos_dir = Path(args.videos_dir)
    videos = scan_videos(videos_dir)
    if not videos:
        print(f"[ERROR] No valid .mp4 found in {videos_dir}", file=sys.stderr)
        sys.exit(1)

    by_id = group_by_identity(videos)
    identities = sorted(by_id.keys(), key=str.lower)
    print(f"Found {len(identities)} identities: {identities}")
    for name, items in by_id.items():
        groups = sorted(set(g for _, g in items))
        print(f"  - {name:<12}  {len(items)} videos  groups={groups}")

    if args.auto:
        registered, stranger, identity_split = auto_split(identities, args.registered, args.stranger, args.seed)
        mode, seed = "auto", args.seed
    elif args.interactive:
        registered, stranger, identity_split = interactive_split(identities)
        mode, seed = "interactive", None
    else:
        registered, stranger, identity_split = from_config_split(Path(args.from_config), identities)
        mode, seed = f"from_config:{args.from_config}", None

    if not registered or not stranger:
        print("[ERROR] Need at least 1 registered and 1 stranger identity", file=sys.stderr)
        sys.exit(1)

    manifest = build_manifest(videos, registered, stranger, identity_split, mode, seed)
    out_path = Path(args.output)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Manifest written to: {out_path.resolve()}")
    print_summary(manifest, videos)


if __name__ == "__main__":
    main()
