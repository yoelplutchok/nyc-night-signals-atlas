#!/bin/bash
# =============================================================================
# PRISM Tmin Download Script
# Downloads daily minimum temperature data for May-September 2021-2023
# Source: Oregon State University PRISM Climate Group
# =============================================================================

BASE_URL="https://data.prism.oregonstate.edu/time_series/us/an/4km/tmin/daily"
BASE_DIR="/Users/yoelplutchok/Desktop/Sleep_ESI_NYC/data/raw/heat_prism"

echo "=============================================="
echo "PRISM Tmin Download Script"
echo "Downloading May-September data for 2021-2023"
echo "Target directory: $BASE_DIR"
echo "=============================================="
echo ""
echo "Total files to download: 459"
echo ""

# Function to get days in a month
get_days() {
    case $1 in
        5) echo 31 ;;  # May
        6) echo 30 ;;  # June
        7) echo 31 ;;  # July
        8) echo 31 ;;  # August
        9) echo 30 ;;  # September
    esac
}

downloaded_files=0
skipped_files=0

for year in 2021 2022 2023; do
    echo "Processing year: $year"
    
    TARGET_DIR="$BASE_DIR/$year"
    mkdir -p "$TARGET_DIR"
    
    for month in 5 6 7 8 9; do
        days=$(get_days $month)
        month_padded=$(printf "%02d" $month)
        
        echo "  Month: $month_padded ($days days)"
        
        day=1
        while [ $day -le $days ]; do
            day_padded=$(printf "%02d" $day)
            date_str="${year}${month_padded}${day_padded}"
            filename="prism_tmin_us_25m_${date_str}.zip"
            url="${BASE_URL}/${year}/${filename}"
            target_file="${TARGET_DIR}/${filename}"
            
            if [ -f "$target_file" ]; then
                echo "    Skipping $filename (already exists)"
                skipped_files=$((skipped_files + 1))
            else
                echo "    Downloading $filename..."
                curl -s -o "$target_file" "$url"
                if [ $? -eq 0 ] && [ -s "$target_file" ]; then
                    downloaded_files=$((downloaded_files + 1))
                else
                    echo "    ERROR: Failed to download $filename"
                    rm -f "$target_file"
                fi
            fi
            day=$((day + 1))
        done
    done
    echo ""
done

echo "=============================================="
echo "Download complete!"
echo "  Downloaded: $downloaded_files files"
echo "  Skipped (existing): $skipped_files files"
echo "=============================================="
