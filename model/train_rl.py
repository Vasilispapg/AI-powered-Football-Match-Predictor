import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import gym
from train_func import extract_features

# Load the CSV data
data = pd.read_csv('filter/output_filtered.csv')
env = gym.make('CartPole-v1')
initial_state = env.reset()


# Define the RL agent model
class RLAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.1, discount_factor=0.99, state_bins=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.build_model()
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Initialize Q-table
        if state_bins is None:
            state_bins = [20] * state_dim  # Default to 20 bins for each state variable
        self.state_bins = state_bins
        self.q_table = np.zeros(tuple(state_bins + [action_dim]))

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(32, activation='relu', input_shape=(self.state_dim,)))  # Specify input shape here
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(self.action_dim, activation='softmax'))
        return model

    def train(self, states, actions, rewards):
        # Convert actions to one-hot encoding
        actions_one_hot = tf.one_hot(actions, depth=self.action_dim)
        # Define the PPO loss and optimizer (you may need to customize this)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        optimizer = optimizers.Adam(learning_rate=0.001)

        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            action_masks = tf.math.multiply(actions_one_hot, logits)
            selected_action_probs = tf.reduce_sum(action_masks, axis=1)
            loss = -tf.reduce_sum(tf.math.log(selected_action_probs) * rewards)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def predict(self, states):
        return np.argmax(self.model(states), axis=1)
    
    def select_action(self, state):
        # Given the current state, select an action using your policy
        # You can use the agent's model to predict the action probabilities

        # Ensure that the state has a consistent shape (e.g., [state_dim])
        state = np.array(state, dtype=np.float32)

        # Reshape the state to match the input shape (1, state_dim)
        state = state.reshape((1, self.state_dim))

        # Use the agent's model to predict action probabilities
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        action_probabilities = self.model.predict(state_tensor)[0]

        # Sample an action from the action probabilities (e.g., using np.random.choice)
        action = np.random.choice(self.action_dim, p=action_probabilities)

        return action

    
    def discretize_state(self, state):
        # Convert the continuous state into a discrete representation
        discrete_state = 0
        for i, value in enumerate(state):
            discrete_state += value * (self.state_bins[i] - 1)  # Adjust for zero-based indexing
        return int(discrete_state)

    def update(self, state, action, reward, next_state):
        # Discretize the state
        discrete_state = self.discretize_state(state)

        # Ensure action is an integer
        action = int(action)

        # Calculate the Q-value for the current state-action pair
        current_q_value = self.q_table[discrete_state][action]

        # Estimate the maximum Q-value for the next state
        max_next_q_value = np.max(self.q_table[self.discretize_state(next_state)])

        # Update the Q-value using the Q-learning formula
        updated_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * max_next_q_value - current_q_value)

        # Update the Q-table
        self.q_table[discrete_state][action] = updated_q_value

    def update_with_new_data(self, states, actions, rewards, next_states, X_new_scaled, y_new_encoded):
        # Update the agent's model with new data
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        
        # Train the model with the new data
        loss = self.train(states, actions, rewards)
        
        # Update the Q-table using Q-learning
        for i in range(len(states)):
            self.update(states[i], actions[i], rewards[i], next_states[i])
        
        return loss

if __name__ == "__main__":
    state_dim = 4  # Update this based on your feature extraction
    action_dim = 2  # Number of actions

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

    # Create the RL agent
    agent = RLAgent(state_dim, action_dim)
    agent.model.summary()

    # Training loop (initial training from dataset)
    num_initial_episodes = 500  # Number of episodes for initial training
    for episode in range(num_initial_episodes):
        state = env.reset()  # Reset the environment to the initial state
        done = False  # A flag indicating if the episode is done

        states, actions, rewards, next_states = [], [], [], []

        while not done:
            # 1. Collect Data: Choose an action based on the current state and policy
            state_data = state[0]  # Extract the NumPy array from the tuple
            state_array = np.array(state_data, dtype=np.float32)
            action = agent.select_action(state_data)

            # 2. Interact with the environment and receive new state and reward
            next_state, reward, done, _,_ = env.step(action)

            # 3. Calculate Rewards and Update Agent
            agent.update(state_array, action, reward, next_state)

            # Transition to the next state
            state = next_state

        # 3. Calculate Rewards and Update Agent
        agent.update_with_new_data(states, actions, rewards, next_states, X_train_scaled, y_outcome_train)

    # Online Learning (continual learning from new data)
    num_online_episodes = 100  # Number of episodes for online learning
    folder = 'matces_detailed_proccessed'  # Update to the full path of your folder
    for episode in range(num_online_episodes):
        # Load and preprocess new data from your source (e.g., CSV)
        random_file = np.random.choice(os.listdir(folder))
        new_data = pd.read_csv(os.path.join(folder, random_file))
        X_new, _ = extract_features(new_data)
        X_new_scaled = scaler.transform(X_new)

        states, actions, rewards, next_states = [], [], [], []
        
        state = env.reset()  # Reset the environment to the initial state
        done = False  # A flag indicating if the episode is done

        while not done:
            # 1. Collect Data: Choose an action based on the current state and policy
            state_data = state[0]  # Extract the NumPy array from the tuple
            state_array = np.array(state_data, dtype=np.float32)
            action = agent.select_action(state_data)

            # 2. Interact with the environment and receive new state and reward
            next_state, reward, done, _,_ = env.step(action)

            # Define a simple reward function (customize this based on your task)
            reward = 1.0 if not done else -1.0

            # Store the collected data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)

            # Transition to the next state
            state = next_state

        # 3. Calculate Rewards and Update Agent (with new data)
        agent.update_with_new_data(states, actions, rewards, next_states, X_new_scaled, y_outcome_train)

    # Evaluate the RL agent on the test set
    y_pred = agent.predict(X_test_scaled)
    accuracy = accuracy_score(y_outcome_test, y_pred)
    report = classification_report(y_outcome_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)
