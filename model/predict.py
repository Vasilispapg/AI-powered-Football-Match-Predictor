import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
import pickle
import os

# Load the CSV data
data = pd.read_csv('performance/data_performance.csv')

# Define the features
match_data = [74.2, 36.8, 60, 40]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform([match_data])

# Load the trained model
model_filename = 'model/model.pkl'

def load_model(model_filename):
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    return model

if __name__ == "__main__":
    if os.path.exists(model_filename):
        loaded_model = load_model(model_filename)
        print("Loaded saved model.")
        # Predict the winner
        home_win_prob = loaded_model.predict(X_scaled)[0][0]
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
