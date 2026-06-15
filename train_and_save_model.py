import pandas as pd
import os
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data
data_path = 'full_blood_donation_data.csv' if os.path.exists('full_blood_donation_data.csv') else 'https://raw.githubusercontent.com/Adhavan1801/Blood-donation-prediction/main/full_blood_donation_data.csv'
df = pd.read_csv(data_path)

columns = ['Recency', 'Frequency', 'Monetary', 'Time', 'Don']
df.columns = columns
df.drop_duplicates(inplace=True)

# 2. Prepare Features and Target
X = df.drop('Don', axis=1)
y = df['Don']

# 3. Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Grid Search CV Decision Tree (Best approach from notebook)
param_grid = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

clf = DecisionTreeClassifier(criterion='gini', random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
print("Training the model...")
grid_search.fit(X_scaled, y)

best_clf = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# 5. Save Model and Scaler
joblib.dump(best_clf, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Model and scaler saved to model.joblib and scaler.joblib")
