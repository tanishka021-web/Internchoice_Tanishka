# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the dataset (Assuming 'abalone.csv' is in the current directory)
# The dataset can be found online, or you may have it locally.
# Columns usually look like: ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

df = pd.read_csv('abalone.csv')

# Step 3: Data exploration
print(df.head())
print(df.info())
print(df.describe())

# Step 4: Data preprocessing
# Map 'Sex' categorical data to numerical values
df['Sex'] = df['Sex'].map({'M': 0, 'F': 1, 'I': 2})  # M: Male, F: Female, I: Infant

# Create a new 'Age' column by adding 1.5 to the 'Rings' column
df['Age'] = df['Rings'] + 1.5

# Step 5: Exploratory data analysis (EDA)
# Plotting distribution of the Age
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Distribution of Abalone Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Step 6: Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Step 7: Split the data into training and testing sets
X = df.drop(['Rings', 'Age'], axis=1)  # All features except 'Rings' and 'Age'
y = df['Age']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Make predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plotting the predicted vs actual ages
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.title('Actual vs Predicted Age')
plt.show()

