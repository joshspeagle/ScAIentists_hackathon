#!/bin/bash
# Monitor enrichment progress

echo "=========================================="
echo "Enrichment Progress Monitor"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

while true; do
    echo "=== Update at $(date '+%H:%M:%S') ==="

    # Count completed locations in japan.csv (should have climate columns)
    if [ -f "data/japan.csv" ]; then
        # Check if file has climate columns
        if head -1 data/japan.csv | grep -q "spring_temp"; then
            completed=$(awk -F',' 'NR>1 && $8!="" {print $1}' data/japan.csv | sort -u | wc -l)
            echo "Japan: $completed locations enriched with climate data"
        else
            echo "Japan: Enrichment in progress (no climate columns yet)"
        fi
    fi

    # Check other files
    for file in data/{kyoto,liestal,meteoswiss,nyc,south_korea,vancouver,washingtondc}.csv; do
        if [ -f "$file" ]; then
            filename=$(basename $file)
            if head -1 "$file" | grep -q "spring_temp"; then
                echo "$filename: ✓ Enriched"
            else
                echo "$filename: Pending"
            fi
        fi
    done

    echo ""
    echo "Checking again in 5 minutes..."
    echo ""

    sleep 300  # 5 minutes
done
