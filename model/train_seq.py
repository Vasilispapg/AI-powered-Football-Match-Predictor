import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.utils import to_categorical
import time
import os
import numpy as np
import tensorflow as tf
from train_func import extract_features


data = pd.read_csv('filter/output_filtered.csv')

def build_and_train_model(X_train_scaled, y_outcome_train):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=X_train_scaled.shape[1]))
    model.add(Dropout(0.2))  # Adding dropout for regularization
    model.add(Dense(16, activation='relu'))
    model.add(Dense(len(y_outcome_train[0]), activation='softmax'))  # Output for outcome classes

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    log_dir = "logs/" + time.strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True)

    model.fit(X_train_scaled, y_outcome_train, epochs=50, batch_size=32,
              validation_split=0.15, callbacks=[tensorboard_callback])
    return model

if __name__ == "__main__":
    model_filename = 'model/model_multi_output.h5'
    
    # Check if the saved model exists
    if os.path.exists(model_filename):
        print("Loaded saved model.")
    else:
        # Extract features and handle missing data
        X, y_outcome_encoded = extract_features(data)

        # Split the data into training and testing sets (80-20 split)
        X_train, X_test, y_outcome_train, y_outcome_test = train_test_split(
            X, y_outcome_encoded, test_size=0.3, random_state=101
        )
        
        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert outcome labels to one-hot encoded format
        y_outcome_train = to_categorical(y_outcome_train)
        y_outcome_test = to_categorical(y_outcome_test)
        
        print("Saved model not found. Building and training a new model.")
        trained_model = build_and_train_model(X_train_scaled, y_outcome_train)

        # Save the model
        trained_model.save(model_filename)
    
    # Load the saved model
    loaded_model = tf.keras.models.load_model(model_filename)

    # Evaluate the model on the test set
    y_pred = np.argmax(loaded_model.predict(X_test_scaled), axis=-1)
    accuracy = accuracy_score(np.argmax(y_outcome_test, axis=-1), y_pred)
    report = classification_report(np.argmax(y_outcome_test, axis=-1), y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)