# Step 1: Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset (assuming 'premier_league_2018_2019.csv' is in the current directory)
df = pd.read_csv('premier_league_2018_2019.csv')

# Step 3: Explore the dataset
print(df.head())
print(df.info())
print(df.describe())

# Step 4: Data Preprocessing
# Check for missing values
print(df.isnull().sum())

# Dropping any rows or columns with missing data if necessary (optional)
# df.dropna(inplace=True)

# Step 5: Analyzing the match results
# Count the total number of home wins, away wins, and draws
home_wins = df[df['FTHG'] > df['FTAG']].shape[0]  # Home Team Goals > Away Team Goals
away_wins = df[df['FTAG'] > df['FTHG']].shape[0]  # Away Team Goals > Home Team Goals
draws = df[df['FTHG'] == df['FTAG']].shape[0]     # Draws

# Plot the distribution of results
results = ['Home Wins', 'Away Wins', 'Draws']
values = [home_wins, away_wins, draws]

plt.figure(figsize=(6, 6))
plt.pie(values, labels=results, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral'])
plt.title('Premier League 2018-2019 Match Outcomes')
plt.show()

# Step 6: Analyzing team statistics
# Top 5 teams with most goals scored (FTHG = Full Time Home Goals, FTAG = Full Time Away Goals)
df['Total Goals Scored'] = df['FTHG'] + df['FTAG']

team_goals = df.groupby('HomeTeam')['Total Goals Scored'].sum().sort_values(ascending=False).head(5)
print("Top 5 Teams by Goals Scored:\n", team_goals)

# Step 7: Visualizing team performance (e.g., goals scored by each team)
plt.figure(figsize=(10, 6))
sns.barplot(x=team_goals.index, y=team_goals.values, palette='viridis')
plt.title('Top 5 Teams by Total Goals Scored (2018-2019)')
plt.ylabel('Goals Scored')
plt.xlabel('Team')
plt.show()

# Step 8: Analyzing individual player performance (optional)
# Assuming you have a 'Players' column with statistics
# Example: Finding top scorers (if player data is included in the dataset)
if 'Players' in df.columns and 'Goals' in df.columns:
    top_scorers = df.groupby('Players')['Goals'].sum().sort_values(ascending=False).head(10)
    print("Top 10 Scorers:\n", top_scorers)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_scorers.index, y=top_scorers.values, palette='rocket')
    plt.title('Top 10 Scorers of Premier League 2018-2019')
    plt.ylabel('Goals')
    plt.xlabel('Players')
    plt.xticks(rotation=90)
    plt.show()

# Step 9: Optional - Predictive Analysis (Machine Learning)
# Example: Predicting match outcomes using basic machine learning models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create a binary target variable: 1 if home team wins, 0 otherwise
df['HomeWin'] = np.where(df['FTHG'] > df['FTAG'], 1, 0)

# Features and target variable
features = ['FTHG', 'FTAG', 'HS', 'AS', 'HC', 'AC']  # Example: Goals, Shots, Corners
X = df[features]
y = df['HomeWin']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict the outcomes for the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the prediction model: {accuracy:.2f}")

# Step 10: Visualize the predicted results (Optional)
# Confusion Matrix can be plotted here to visualize how well the model is performing
