import pandas as pd
import numpy as np
import warnings  # Import the warnings module
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Suppress specific sklearn warning
warnings.filterwarnings("ignore", category=UserWarning, message="X has feature names")

# Load dataset
data = pd.read_csv("material_losses.csv")

# Separate features and target
X = data.drop('loss', axis=1)
y = data['loss']

# Encode categorical features
categorical_features = ['material_type']
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Corrected parameter name
encoded_features = encoder.fit_transform(X[categorical_features])

# Combine encoded features with other features (only distance remains)
X_encoded = np.hstack((X.drop(categorical_features, axis=1).values, encoded_features))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Define the Gradient Boosting Regressor
model = GradientBoostingRegressor(random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Set up GridSearchCV with limited parallel processing (n_jobs=1 for no parallelism)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=1)

try:
    # Fit the model using grid search
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")

    # Optionally evaluate the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²): {r2}")

except Exception as e:
    print(f"An error occurred: {e}")
