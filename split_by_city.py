#!/usr/bin/env python3
"""
Split country-level CSV files into city-level CSV files.
"""

import csv
import os
from collections import defaultdict

def split_csv_by_city(input_file, output_dir='./data'):
    """
    Split a CSV file by city based on the 'location' column.

    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save the output files
    """
    # Read the CSV file and group by city
    city_data = defaultdict(list)
    header = None

    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames

        for row in reader:
            # Extract city from location (format: "Country/City")
            location = row['location']
            city = location.split('/')[-1] if '/' in location else location
            city_data[city].append(row)

    output_files = []

    # Write each city's data to a separate file
    for city, rows in city_data.items():
        # Create filename from city name (lowercase, replace spaces with underscores)
        city_filename = city.lower().replace(' ', '_').replace('-', '_')
        output_file = os.path.join(output_dir, f'{city_filename}.csv')

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)

        output_files.append(output_file)
        print(f"Created {output_file} with {len(rows)} records")

    return output_files

if __name__ == '__main__':
    import sys

    # Files to split
    files_to_split = ['./data/japan.csv', './data/south_korea.csv']

    all_output_files = []

    for input_file in files_to_split:
        if os.path.exists(input_file):
            print(f"\nProcessing {input_file}...")
            output_files = split_csv_by_city(input_file)
            all_output_files.extend(output_files)
        else:
            print(f"Warning: {input_file} not found, skipping...")

    print(f"\n✓ Successfully created {len(all_output_files)} city-level CSV files")
