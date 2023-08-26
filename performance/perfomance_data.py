import csv
from collections import defaultdict
import math
import numpy as np

# Directory containing the input CSV file
input_csv_file = "filter/output_filtered.csv"

# Name of the output CSV file
output_csv_file = "performance/data_performance.csv"

# Calculate Win Probability
# win_count: Number of wins in recent matches
# total_recent_matches: Total number of recent matches
# mv_difference_normalized: Normalized difference in market values (MV Home Team - MV Away Team)
def calculate_win_probability(mv_difference_normalized,home_winning_form,away_winning_form):
    # Sigmoid function parameters
    k = 0.3  # Controls the steepness of the curve
    x0 = 1   # Midpoint of the sigmoid
    combined_input = mv_difference_normalized * home_winning_form / away_winning_form if (away_winning_form!=0) else mv_difference_normalized * home_winning_form / 0.01

    probability = 1 / (1 + math.exp(-k * (combined_input - x0)))
    return probability


rolling_window_size = 5


# Read the input CSV and calculate team performance
team_matches = defaultdict(list)
keep_data= []
with open(input_csv_file, "r", newline="", encoding="utf-8") as input_csv:
    reader = csv.DictReader(input_csv)
    for row in reader:
        if len(row) == 9 and all(row.values()):
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
                    team_matches[home_team].append(("win",'loss', date,mv_home_team,mv_away_team,home_team,away_team))
                    team_matches[away_team].append(("loss",'win', date,mv_home_team,mv_away_team,home_team,away_team))
                elif home_score < away_score:
                    team_matches[home_team].append(("loss",'win', date,mv_home_team,mv_away_team,home_team,away_team))
                    team_matches[away_team].append(("win",'loss', date,mv_home_team,mv_away_team,home_team,away_team))
                else:
                    team_matches[home_team].append(("draw",'draw', date,mv_home_team,mv_away_team,home_team,away_team))
                    team_matches[away_team].append(("draw",'draw', date,mv_home_team,mv_away_team,home_team,away_team))
            keep_data.append(row) #in order to delete the empty rows


    team_probabilities = {}
    for team, matches in team_matches.items():
        team_probabilities[team] = []
        # Calculate winning form for each match
        for i, (outcome_home,outcome_away, match_date, mv_home_team, mv_away_team, home_team, away_team) in enumerate(matches):
            
            # Calculate home team's winning form for the same matches
            recent_matches_home = []
            for j in range(i, max(i - 10, -1), -1):
                if(matches[j][5]==home_team):
                    recent_matches_home.append(matches[j][0])
                elif(matches[j][6]==home_team):
                     recent_matches_home.append(matches[j][1])

            home_win_count = recent_matches_home.count("win")
            home_winning_form = home_win_count / len(recent_matches_home) if len(recent_matches_home) > 0 else 0
            
            recent_matches_away=[]
            # Calculate away team's winning form for the same matches
            for j in range(i, max(i - 10, -1), -1):
                    if(matches[j][5]==away_team):
                        recent_matches_away.append(matches[j][0])
                    elif(matches[j][6]==away_team):
                        recent_matches_away.append(matches[j][1])
            away_win_count = recent_matches_away.count("win")
            away_winning_form = away_win_count / len(recent_matches_away) if len(recent_matches_away) > 0 else 0

            # Append winning form to the match data
            matches[i] = (outcome_home,outcome_away, match_date, mv_home_team, mv_away_team, home_team, away_team, home_winning_form, away_winning_form)

    # Calculate probabilities using winning form for both teams
    for team, matches in team_matches.items():
        # Sort matches by date in ascending order
        matches.sort(key=lambda x: x[1])

        for i, (outcome_home,outcome_away, match_date, mv_home_team, mv_away_team, home_team, away_team, home_winning_form, away_winning_form) in enumerate(matches):
            mv_difference_normalized = (mv_home_team - mv_away_team) / 100

            home_win_probability = calculate_win_probability(mv_difference_normalized, home_winning_form,away_winning_form)
            away_win_probability = 1-home_win_probability
            team_probabilities[home_team].append((home_win_probability, match_date))
            team_probabilities[away_team].append((away_win_probability, match_date))
           

# Write the calculated probabilities to the output CSV
with open(output_csv_file, "w", newline="", encoding="utf-8") as output_csv:
    fieldnames = ["Competition", "Country", "Home Team", "Home Score", "Away Team", "Away Score",'MV Home Team', 'MV Away Team' ,"Date", "Home_Win_Probability", "Away_Win_Probability"]
    writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
    writer.writeheader()

    for row in keep_data:
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
