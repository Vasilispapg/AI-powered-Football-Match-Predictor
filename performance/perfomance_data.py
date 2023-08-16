import csv
from collections import defaultdict
import math
import numpy as np

# Directory containing the input CSV file
input_csv_file = "filter/output_filtered.csv"

# Name of the output CSV file
output_csv_file = "performance/data_performance.csv"

# Read the input CSV and calculate team performance
team_matches = defaultdict(list)
with open(input_csv_file, "r", newline="", encoding="utf-8") as input_csv:
    reader = csv.DictReader(input_csv)
    for row in reader:
        if len(row) == 9 and None not in row.values():
            home_team = row["Home Team"]
            away_team = row["Away Team"]
            home_score = row["Home Score"]
            away_score = row["Away Score"]
            date = row["Date"]  # Get the Date value as a string
            mv_home_team = float(row.get("MV Home Team", 1.0))  # Default value of 1.0 if not provided
            mv_away_team = float(row.get("MV Away Team", 1.0))  # Default value of 1.0 if not provided

            if home_score and home_score.isdigit() and away_score and away_score.isdigit():
                home_score = int(home_score)
                away_score = int(away_score)

                if home_score > away_score:
                    team_matches[home_team].append(("win", date,mv_home_team,mv_away_team))
                    team_matches[away_team].append(("loss", date,mv_home_team,mv_away_team))
                elif home_score < away_score:
                    team_matches[home_team].append(("loss", date,mv_home_team,mv_away_team))
                    team_matches[away_team].append(("win", date,mv_home_team,mv_away_team))
                else:
                    team_matches[home_team].append(("draw", date,mv_home_team,mv_away_team))
                    team_matches[away_team].append(("draw", date,mv_home_team,mv_away_team))


team_probabilities = {}
for team, matches in team_matches.items():

    team_probabilities[team] = []

    # Sort matches by date in ascending order
    matches.sort(key=lambda x: x[1])

    for i, (outcome, match_date, mv_home_team, mv_away_team) in enumerate(matches):
        # Find the most recent 5 matches before the current match's date
        recent_matches = [matches[j][0] for j in range(i, max(i-10, -1), -1)]
        win_count = recent_matches.count("win")
        total_recent_matches = len(recent_matches)

        # Calculate win probability using Laplace smoothing
        win_probability = (win_count + 1) / (total_recent_matches + 3)

        if mv_home_team > mv_away_team:
            weighted_win_probability = win_probability + (mv_home_team - mv_away_team) / 100
        else:
            weighted_win_probability = win_probability


        team_probabilities[team].append((weighted_win_probability, match_date))


# Write the calculated probabilities to the output CSV
with open(output_csv_file, "w", newline="", encoding="utf-8") as output_csv:
    fieldnames = ["Competition", "Country", "Home Team", "Home Score", "Away Team", "Away Score",'MV Home Team', 'MV Away Team' ,"Date", "Home_Win_Probability", "Away_Win_Probability"]
    writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
    writer.writeheader()

    with open(input_csv_file, "r", newline="", encoding="utf-8") as input_csv:
        reader = csv.DictReader(input_csv)
        for row in reader:
            home_team = row["Home Team"]
            away_team = row["Away Team"]
            date=row['Date']

            # Find the matching probability for the current match's date
            home_win_probability = round(next((prob for prob, match_date in team_probabilities.get(home_team, []) if match_date == date), 0)*100,3)
            away_win_probability = round(next((prob for prob, match_date in team_probabilities.get(away_team, []) if match_date == date), 0)*100,3)

            # Update the row dictionary with the calculated values
            row.update({"Home_Win_Probability": home_win_probability, "Away_Win_Probability": away_win_probability})
            writer.writerow(row)

print("Win probability data added and saved to:", output_csv_file)
