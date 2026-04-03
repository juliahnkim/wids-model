#!/bin/bash
# Download Census TIGER/Line Roads shapefiles for all US counties (50 states + DC)
# Files: tl_2024_{SSCCC}_roads.zip
# Source: https://www2.census.gov/geo/tiger/TIGER2024/ROADS/

DEST="data/raw/tiger_roads"
URL_FILE="$DEST/download_urls.txt"
MAX_PARALLEL=20
RETRY=3

echo "Downloading TIGER roads to $DEST ..."
echo "URLs: $(wc -l < "$URL_FILE")"
echo "Parallel connections: $MAX_PARALLEL"

# Download function — skip if already exists
download_one() {
    url="$1"
    fname=$(basename "$url")
    dest="data/raw/tiger_roads/$fname"
    if [ -f "$dest" ] && [ -s "$dest" ]; then
        return 0
    fi
    curl -sS --retry "$RETRY" --retry-delay 2 -o "$dest" "$url" 2>/dev/null
}
export -f download_one

# Run parallel downloads
cat "$URL_FILE" | xargs -P "$MAX_PARALLEL" -I{} bash -c 'download_one "{}"'

# Count results
total=$(ls "$DEST"/tl_2024_*_roads.zip 2>/dev/null | wc -l)
echo "Downloaded: $total ZIP files"
