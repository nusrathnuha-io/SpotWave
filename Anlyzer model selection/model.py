import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("material_losses.csv")

# Separate features and target
X = data.drop('loss', axis=1)
y = data['loss']

# Encode categorical features
categorical_features = ['material_type']
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(X[categorical_features])

# Combine encoded features with other features (only distance remains)
X_encoded = np.hstack((X.drop(categorical_features, axis=1).values, encoded_features))

# Normalize the features
scaler = StandardScaler()
X_encoded = scaler.fit_transform(X_encoded)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Define and train the Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.2, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and encoder
joblib.dump(model, 'gradient_boosting_model.pkl')
joblib.dump(encoder, 'material_type_encoder.pkl')

# Optionally, save feature names for future reference
feature_names = X.drop(categorical_features, axis=1).columns.tolist() + encoder.get_feature_names_out(categorical_features).tolist()
joblib.dump(feature_names, 'feature_names.pkl')

# Evaluate the model
y_pred = model.predict(X_test)

# Calculate MSE, R², MAE, MedAE, and RMSE
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Median Absolute Error (MedAE): {medae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Optional: Log the results
with open('model_evaluation.txt', 'w') as f:
    f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
    f.write(f"R-squared (R²): {r2:.4f}\n")
    f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
    f.write(f"Median Absolute Error (MedAE): {medae:.4f}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")

# Visualizations
figsize = (6, 4)

# Scatter plot of actual vs predicted values
plt.figure(figsize=figsize)
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
plt.title('Scatter plot')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.tight_layout()
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=figsize)
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title('Residual plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.tight_layout()
plt.show()

# Histogram of residuals
plt.figure(figsize=figsize)
sns.histplot(residuals, kde=True, bins=30)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()
