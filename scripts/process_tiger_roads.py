"""
Process Census TIGER/Line Roads shapefiles into county-level lane miles.

Reads all tl_2024_SSCCC_roads.zip from data/raw/tiger_roads/,
filters to major road classes (S1100, S1200, S1400),
computes lane miles per county per MTFCC code,
and writes data/processed/tiger_road_miles_by_county.csv.

MTFCC → Functional Class mapping:
  S1100  Primary Road      (Interstate / Principal Arterial)  — assumed 4 lanes
  S1200  Secondary Road    (State Hwy / Minor Arterial)       — assumed 2 lanes
  S1400  Local Neighborhood Road                              — assumed 2 lanes

Lane miles = road segment length (miles) × assumed lane count.
"""

import glob
import os
import sys
import zipfile
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Check for geopandas / fiona
try:
    import geopandas as gpd
except ImportError:
    print("ERROR: geopandas required. Install: pip install geopandas")
    sys.exit(1)

# MTFCC codes we care about and their assumed lane counts
MTFCC_LANES = {
    "S1100": 4,   # Primary Road (Interstate / Principal Arterial)
    "S1200": 2,   # Secondary Road (State Hwy / Minor Arterial)
    "S1400": 2,   # Local Neighborhood Road
}

TIGER_DIR = Path("data/raw/tiger_roads")
OUTPUT_PATH = Path("data/processed/tiger_road_miles_by_county.csv")

METERS_PER_MILE = 1609.344


def process_one_zip(zip_path: str) -> pd.DataFrame:
    """
    Read a single TIGER roads ZIP, filter to S1100/S1200/S1400,
    compute lane miles per MTFCC.
    Returns DataFrame with columns: fips, mtfcc, lane_miles
    """
    # Extract FIPS from filename: tl_2024_SSCCC_roads.zip
    fname = os.path.basename(zip_path)
    fips_str = fname.split("_")[2]  # SSCCC
    fips = int(fips_str)

    try:
        gdf = gpd.read_file(f"zip://{zip_path}")
    except Exception as e:
        print(f"  WARNING: Failed to read {fname}: {e}")
        return pd.DataFrame(columns=["fips", "mtfcc", "lane_miles"])

    if gdf.empty or "MTFCC" not in gdf.columns:
        return pd.DataFrame(columns=["fips", "mtfcc", "lane_miles"])

    # Filter to road classes we care about
    gdf = gdf[gdf["MTFCC"].isin(MTFCC_LANES.keys())].copy()
    if gdf.empty:
        return pd.DataFrame(columns=["fips", "mtfcc", "lane_miles"])

    # Project to equal-area for accurate length calculation
    # Use Albers Equal Area (EPSG:5070) for CONUS
    state_fips = fips_str[:2]
    if state_fips == "02":
        # Alaska — use Alaska Albers (EPSG:3338)
        gdf_proj = gdf.to_crs(epsg=3338)
    elif state_fips == "15":
        # Hawaii — use Hawaii State Plane (EPSG:26963)
        gdf_proj = gdf.to_crs(epsg=26963)
    else:
        gdf_proj = gdf.to_crs(epsg=5070)

    # Compute road segment length in miles
    gdf_proj["length_miles"] = gdf_proj.geometry.length / METERS_PER_MILE

    # Compute lane miles
    gdf_proj["lane_count"] = gdf_proj["MTFCC"].map(MTFCC_LANES)
    gdf_proj["lane_miles"] = gdf_proj["length_miles"] * gdf_proj["lane_count"]

    # Aggregate by MTFCC
    result = (
        gdf_proj.groupby("MTFCC")["lane_miles"]
        .sum()
        .reset_index()
        .rename(columns={"MTFCC": "mtfcc"})
    )
    result["fips"] = fips

    return result[["fips", "mtfcc", "lane_miles"]]


def main():
    print("Processing Census TIGER/Line Roads shapefiles...")
    print(f"Source: {TIGER_DIR}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    zip_files = sorted(glob.glob(str(TIGER_DIR / "tl_2024_*_roads.zip")))
    print(f"ZIP files found: {len(zip_files)}")

    all_results = []
    errors = 0
    for i, zf in enumerate(zip_files, 1):
        if i % 500 == 0 or i == len(zip_files):
            print(f"  Processed {i}/{len(zip_files)} counties...")
        try:
            result = process_one_zip(zf)
            if not result.empty:
                all_results.append(result)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR on {os.path.basename(zf)}: {e}")

    if not all_results:
        print("ERROR: No results produced. Check TIGER files.")
        sys.exit(1)

    df = pd.concat(all_results, ignore_index=True)
    df["fips"] = df["fips"].astype(int)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")

    # ── Diagnostics ──
    print(f"\n{'='*60}")
    print("DIAGNOSTICS")
    print(f"{'='*60}")

    counties = df["fips"].nunique()
    print(f"Total counties in output: {counties}")

    print("\nLane miles distribution per MTFCC class:")
    for mtfcc in ["S1100", "S1200", "S1400"]:
        sub = df[df["mtfcc"] == mtfcc]
        if sub.empty:
            print(f"  {mtfcc}: no data")
        else:
            print(f"  {mtfcc} ({MTFCC_LANES[mtfcc]}-lane): "
                  f"counties={len(sub)}, "
                  f"mean={sub['lane_miles'].mean():.1f}, "
                  f"median={sub['lane_miles'].median():.1f}, "
                  f"max={sub['lane_miles'].max():.1f}, "
                  f"total={sub['lane_miles'].sum():.0f}")

    # Counties missing a road class
    all_fips = set(df["fips"].unique())
    for mtfcc in ["S1100", "S1200", "S1400"]:
        fips_with = set(df[df["mtfcc"] == mtfcc]["fips"].unique())
        missing = all_fips - fips_with
        if missing:
            print(f"\n  Counties with ZERO {mtfcc} roads: {len(missing)}")
            if len(missing) <= 10:
                print(f"    FIPS: {sorted(missing)}")
        else:
            print(f"\n  Counties with ZERO {mtfcc} roads: 0")

    if errors > 0:
        print(f"\n  Errors during processing: {errors}")


if __name__ == "__main__":
    main()
