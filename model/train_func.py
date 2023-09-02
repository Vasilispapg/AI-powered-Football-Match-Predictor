import pandas as pd
import ast

def encode_match_outcome(home_score, away_score):
    """
    Encode the match outcome as 'Home Team Win,' 'Draw,' or 'Away Team Win'
    based on the final scores of the home and away teams.
    """
    if home_score > away_score:
        return 'Home Team Win'
    elif home_score < away_score:
        return 'Away Team Win'
    else:
        return 'Draw'

def extract_features(data):
    """
    Extract and preprocess features from the 'Votes' and 'Stats' columns.
    This function can be customized based on your data structure.
    """
    X_features = []
    y_labels = []

    for index, row in data.iterrows():
        try:
            
            # Initialize variables for stats
            ball_possession_home = None
            ball_possession_away = None
            total_shots_home = None
            total_shots_away = None
            corners_home = None
            corners_away = None
            fouls_home = None
            fouls_away = None
            votes_home = None
            votes_draw = None
            votes_away = None
            votes_difference = None
            
            # Extract features from the 'Votes' column  
            votes = eval(row['Votes'])
            if votes:
                votes_home = int(votes[1][0])
                votes_draw = int(votes[1][1])
                votes_away = int(votes[1][2])
                votes_difference = votes_home - votes_away
                

                # Extract features from the 'Stats' column
            if 'Stats' in row and not pd.isna(row['Stats']):
                # Use ast.literal_eval to safely evaluate a Python literal expression
                stats = ast.literal_eval(row['Stats'])
                
                # Check if 'Ball possession' key is present and not empty
                if 'Ball possession' in stats and stats['Ball possession']:
                    ball_possession_home = round(float(stats['Ball possession'][0][:-1]) / 100, 3)
                    ball_possession_away = round(float(stats['Ball possession'][1][:-1]) / 100, 3)

                # Check if other keys are present and not empty (similar to 'Ball possession')
                if 'Total shots' in stats and stats['Total shots']:
                    total_shots_home = int(stats['Total shots'][0])
                    total_shots_away = int(stats['Total shots'][1])
                
                if 'Corners' in stats and stats['Corners']:
                    corners_home = int(stats['Corners'][0])
                    corners_away = int(stats['Corners'][1])
                
                if 'Fouls' in stats and stats['Fouls']:
                    fouls_home = int(stats['Fouls'][0])
                    fouls_away = int(stats['Fouls'][1])

            # Extract other relevant features
            mv_home_team = row['MV Home Team']
            mv_away_team = row['MV Away Team']
            
            # Define the feature vector for this match
            features = [
                mv_home_team, mv_away_team, votes_home, votes_draw, votes_away, votes_difference,
                ball_possession_home, ball_possession_away, total_shots_home, total_shots_away,
                corners_home, corners_away, fouls_home, fouls_away
            ]

            # Append the feature vector and label to the lists
            X_features.append(features)
            y_labels.append(row['Home Score'] > row['Away Score'])

        except KeyError as ke:
            # Handle KeyError
            print(ball_possession_home, ball_possession_away, total_shots_home, total_shots_away, corners_home, corners_away, fouls_home, fouls_away)
            print(f"KeyError at index {index}: {ke}")
        except ValueError as ve:
            # Handle ValueError
            print(f"ValueError at index {index}: {ve}")
        except TypeError as te:
            # Handle TypeError
            print(f"TypeError at index {index}: {te}")
        except Exception as e:
            # Handle other exceptions
            print(f"An unexpected error occurred at index {index}: {e}")

    return X_features, y_labels

