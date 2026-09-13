import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import os

# Load the CSV data
data = pd.read_csv('performance/data_performance.csv')

# Define the features you want to use for prediction
selected_features = [
    'mv_home_team', 'mv_away_team', 'votes_home', 'votes_draw','home_team','away_team'
    # Add other features you want to use here
]

# Standardize the selected features
scaler = StandardScaler()
X_scaled = scaler.fit_transform([selected_features])

# Load the trained model
model_filename = 'model/model_multi_output.h5'

if __name__ == "__main__":
    if os.path.exists(model_filename):
        loaded_model = load_model(model_filename)
        print("Loaded saved model.")
        
        # Ensure that selected_features has the same number of features as the model expects
        if len(selected_features) != len(loaded_model.input_shape[1]):
            print("Input data should have the same number of features as the model's input shape.")
        else:
            # Predict the winner using the selected features
            home_win_prob = loaded_model.predict([selected_features])[0][0]
            away_win_prob = 1 - home_win_prob
            
            print(f"Home win probability: {home_win_prob}")
            print(f"Away win probability: {away_win_prob}")
            
            if home_win_prob > away_win_prob:
                predicted_winner = "Home Team"
            elif home_win_prob < away_win_prob:
                predicted_winner = "Away Team"
            else:
                predicted_winner = "Draw"
            
            print(f"The predicted winner is: {predicted_winner}")
    else:
        print("Model not found. Please train the model first.")
