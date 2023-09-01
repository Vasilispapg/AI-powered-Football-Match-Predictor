import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import time
import pickle
import os

# Load the CSV data
data = pd.read_csv('performance/data_performance.csv')

# Define the features and target variable
X = data[['MV Home Team', 'MV Away Team', 'Home_Win_Probability', 'Away_Win_Probability']]
y = data['Home Score'] > data['Away Score']

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train the neural network model
def build_and_train_model():
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=X_train_scaled.shape[1]))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    log_dir = "logs/" + time.strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True)

    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.15, callbacks=[tensorboard_callback])
    return model

# Save the model to a file
def save_model(model, model_filename):
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    model_filename = 'model/model.pkl'
    
    # Check if the saved model exists
    if os.path.exists(model_filename):
        print("Loaded saved model.")
    else:
        print("Saved model not found. Building and training a new model.")
        trained_model = build_and_train_model()
        save_model(trained_model, model_filename)
