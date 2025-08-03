# Step 1: Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the Data
df = pd.read_csv('D:\coding\python\Task3\IRIS.csv')
print(df.head())

# Step 3: Check for Nulls and Data Info
print(df.info())
print(df.describe())

# Step 4: Encode the Target Label
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Step 5: Split Features and Target
X = df.drop('species', axis=1)
y = df['species']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train Logistic Regression Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 8: Make Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 10: Visualize with Pairplot
sns.pairplot(df, hue='species', palette='Dark2')
plt.show()
