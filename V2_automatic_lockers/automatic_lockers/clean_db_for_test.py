"""Remove identities from the FaceNet DB that should NOT be in the recognition
test as registered users.

For the 8R + 5S split:
  Registered (keep): Diep, Huy, Khai, Mai, Minh, Phuong, TAnh, Viet
  Stranger  (remove from DB if present): An, Bac, Doanh, Hoa, Tai
  Leftover  (remove from DB if present): Huong, Huong... (any name not in the
            manifest's registered_identities list)

Usage:
    python clean_db_for_test.py --manifest recognition_split.json
    python clean_db_for_test.py --manifest recognition_split.json --dry_run

This script connects directly to the SQLite DB via app.get_conn(). It only
deletes from the `faces` table, never from `lockers` or `access_logs`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


# ----------------------------------------------------------------------
# Locate and import app.py from the current working directory
# ----------------------------------------------------------------------


def import_app():
    cwd = Path.cwd()
    app_path = cwd / "app.py"
    if not app_path.is_file():
        print(
            "[ERROR] app.py not found. Run this script from the project root "
            "(the folder that contains app.py).",
            file=sys.stderr,
        )
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("app", str(app_path))
    app = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(app)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Could not import app.py: {exc}", file=sys.stderr)
        sys.exit(1)
    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="recognition_split.json")
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print which entries would be deleted, do not modify DB",
    )
    p.add_argument(
        "--extra_keep",
        nargs="*",
        default=[],
        help="Extra usernames to keep (besides registered_identities)",
    )
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"[ERROR] Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    keep = set(manifest.get("registered_identities", []))
    keep.update(args.extra_keep)
    if not keep:
        print("[ERROR] No registered_identities in manifest. Aborting.", file=sys.stderr)
        sys.exit(1)

    app_mod = import_app()
    get_conn = getattr(app_mod, "get_conn", None)
    refresh_cache = getattr(app_mod, "_refresh_cache", None)
    if get_conn is None:
        print("[ERROR] app.py must expose get_conn()", file=sys.stderr)
        sys.exit(1)

    conn = get_conn()
    rows = conn.execute("SELECT username FROM faces").fetchall()
    all_users = [r[0] for r in rows]
    conn.close()

    to_delete = [u for u in all_users if u not in keep]

    print("\n=== DB cleanup plan ===")
    print(f"Keep ({len(keep)}): {sorted(keep)}")
    print(f"DB currently contains ({len(all_users)}): {all_users}")
    if not to_delete:
        print("\n[OK] DB is already clean. Nothing to delete.")
        return

    print(f"\nWill delete ({len(to_delete)}): {to_delete}")
    if args.dry_run:
        print("\n[DRY RUN] No changes made. Re-run without --dry_run to apply.")
        return

    conn = get_conn()
    for username in to_delete:
        conn.execute("DELETE FROM faces WHERE username=?", (username,))
        print(f"  [DEL] {username}")
    conn.commit()
    conn.close()

    if refresh_cache:
        refresh_cache()

    # Verify
    conn = get_conn()
    rows = conn.execute("SELECT username FROM faces").fetchall()
    conn.close()
    final = [r[0] for r in rows]
    print(f"\n[OK] DB now contains ({len(final)}): {final}")


if __name__ == "__main__":
    main()
