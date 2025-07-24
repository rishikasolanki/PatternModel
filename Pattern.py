# Import necessary libraries	     	                              		#dictionary, maping??
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
url = "https://raw.githubusercontent.com/sachinmotwani20/NPTEL-ML_Datasets/main/InterceptedSignal.csv"
data = pd.read_csv(url)

print("Info\n",data.info())
print("Describe\n",data.describe)
# print(data[species].describe())
print(data["species"].describe())
# print(data[species].value_counts())


# Check for missing values
print("\nMissing values:\n", data.isnull().sum())

# Drop rows with any NaN values
data.dropna(inplace=True)

# Extract features and target
x = data.iloc[:, 0].values.reshape(-1, 1)  # Time
y = data.iloc[:, 1].values.reshape(-1, 1)  # Signal

# Polynomial transformation (degree 5 is good for curves)
poly = PolynomialFeatures(degree=5)
x_poly = poly.fit_transform(x)

# Train the model
model = LinearRegression()
model.fit(x_poly, y)

# Make predictions
y_pred = model.predict(x_poly)

# Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.scatter(x, y, label="Actual Signal", color="blue")
plt.plot(x, y_pred, label="Predicted Signal Trend", color="red")
plt.title("Polynomial Regression on Intercepted Signal")
plt.xlabel("Time (s)")
plt.ylabel("Signal Value")
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nMean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
