"""
Resumable TIGER/Line Roads downloader.

Downloads Census TIGER/Line road shapefiles for all US counties.
Can be restarted — skips already-downloaded valid ZIPs.
Uses delays between requests to avoid Census rate limiting.

Usage:
    python3 scripts/download_tiger_roads.py
    # Ctrl-C to stop, re-run to resume
"""

import os
import time
import subprocess
import pandas as pd
from pathlib import Path

BASE_URL = "https://www2.census.gov/geo/tiger/TIGER2024/ROADS"
DEST = Path("data/raw/tiger_roads")
DELAY = 0.5  # seconds between downloads
MAX_TIME = 60  # curl timeout per file

def get_all_fips():
    df = pd.read_csv("data/raw/national_county2020.txt", sep="|", dtype=str)
    df["fips5"] = df["STATEFP"] + df["COUNTYFP"]
    valid_states = set()
    for i in range(1, 57):
        s = str(i).zfill(2)
        if s not in {"03", "07", "14", "43", "52"}:
            valid_states.add(s)
    df = df[df["STATEFP"].isin(valid_states)]
    return sorted(df["fips5"].tolist())

def is_valid_zip(path):
    """Check if file is a real ZIP archive."""
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        return False
    result = subprocess.run(["file", path], capture_output=True, text=True)
    return "Zip archive" in result.stdout

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    all_fips = get_all_fips()

    # Find what's missing
    missing = []
    for fips in all_fips:
        path = DEST / f"tl_2024_{fips}_roads.zip"
        if not is_valid_zip(str(path)):
            missing.append(fips)

    print(f"Total counties: {len(all_fips)}")
    print(f"Already downloaded: {len(all_fips) - len(missing)}")
    print(f"Remaining: {len(missing)}")

    if not missing:
        print("All files downloaded!")
        return

    errors = 0
    for i, fips in enumerate(missing, 1):
        url = f"{BASE_URL}/tl_2024_{fips}_roads.zip"
        dest = DEST / f"tl_2024_{fips}_roads.zip"

        result = subprocess.run(
            ["curl", "-sS",
             "-A", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
             "--retry", "2", "--retry-delay", "5",
             "--max-time", str(MAX_TIME),
             "--connect-timeout", "10",
             "-o", str(dest), url],
            capture_output=True, text=True
        )

        if is_valid_zip(str(dest)):
            if i % 100 == 0:
                print(f"  {i}/{len(missing)} downloaded...")
        else:
            errors += 1
            if dest.exists():
                dest.unlink()
            if errors <= 3:
                print(f"  Failed: {fips} (will retry on next run)")
            elif errors == 4:
                print(f"  ... suppressing further error messages")

        time.sleep(DELAY)

    # Final count
    valid = sum(1 for fips in all_fips if is_valid_zip(str(DEST / f"tl_2024_{fips}_roads.zip")))
    print(f"\nValid downloads: {valid}/{len(all_fips)}")
    if valid < len(all_fips):
        print(f"Run again to retry {len(all_fips) - valid} missing files.")

if __name__ == "__main__":
    main()
