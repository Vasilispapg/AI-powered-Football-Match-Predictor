import csv

# Path to the input CSV file
input_csv_file = "merged/merged_output_detailed.csv"

# Path to the output CSV file
output_csv_file = "filter/output_filtered.csv"

# Read the input CSV and filter rows
filtered_rows = []
with open(input_csv_file, "r", newline="", encoding="utf-8") as input_csv:
    reader = csv.DictReader(input_csv)
    for row in reader:
        competition = row["Competition"]
        if('cup' in competition.lower()):
            row['Competition_type'] = 'cup'
        elif('friendly' in competition.lower()):
            row['Competition_type'] = 'friendly'
        else:
            row['Competition_type'] = 'league'
        
        if(any(value.lower() == "n/a" for value in row.values())):
            continue
        elif ('u20' not in competition.lower() and 'u19' not in competition.lower() and 'u18' not in competition.lower() and 'u17' not in competition.lower() and 'u21' not in competition.lower() and "women" not in competition.lower()):
            filtered_rows.append(row)
        

# Write the filtered rows to the output CSV file
fieldnames = filtered_rows[0].keys()
with open(output_csv_file, "w", newline="", encoding="utf-8") as output_csv:
    writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(filtered_rows)

print("Filtered data saved to:", output_csv_file)
