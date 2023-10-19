
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('preprocessed_data.csv')

# Convert 'Price' column to numeric, converting non-numeric values to NaN
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Drop rows with NaN values in the 'Price' column
df.dropna(subset=['Price'], inplace=True)
df.dropna(inplace=True)

# Exclude 'Price' column from the features (X)
X = df.drop(columns=['Price'])

# Target variable (y) is 'Price'
y = df['Price']

# Convert categorical variables to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

actual_vs_predicted = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(actual_vs_predicted)

# 1) L1 regularization
# 2) One-hot or any other encoding
# 3) L2 regularizations
# 4) Elastic Net
# 5) Variation and correlation amongst the data
# 6) Usage of the range values instead of the avg for temp, humidity etc

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('preprocessed_data.csv')

# Split the data into training and testing sets (if not already done)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Regressor model
dt_model = DecisionTreeRegressor(random_state=42)

# Train the Decision Tree model on the training data
dt_model.fit(X_train, y_train)

# Make predictions using the Decision Tree model
y_pred_dt = dt_model.predict(X_test)

# Evaluate the Decision Tree model
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = mse_dt**0.5
r2_dt = r2_score(y_test, y_pred_dt)

print("Decision Tree Model:")
print(f"Mean Squared Error (MSE): {mse_dt}")
print(f"Root Mean Squared Error (RMSE): {rmse_dt}")
print(f"R-squared (R2): {r2_dt}")

actual_vs_predicted = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(actual_vs_predicted)

from sklearn.linear_model import Ridge

# Initialize Ridge Regression model with higher alpha for stronger regularization
ridge_alpha = 20 # Adjust the alpha as needed
ridge_model = Ridge(alpha=ridge_alpha)

# Train the Ridge model
ridge_model.fit(X_train, y_train)

# Make predictions using the Ridge model
y_pred_ridge = ridge_model.predict(X_test)

# Evaluate the Ridge model
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = mse_ridge ** 0.5
r2_ridge = r2_score(y_test, y_pred_ridge)

print("Ridge Regression Model:")
print(f"Mean Squared Error (MSE): {mse_ridge}")
print(f"Root Mean Squared Error (RMSE): {rmse_ridge}")
print(f"R-squared (R2): {r2_ridge}")

from sklearn.linear_model import Lasso

lasso_alpha = 20  # Adjust the alpha as needed
lasso_model = Lasso(alpha=lasso_alpha)

# Train the Lasso model
lasso_model.fit(X_train, y_train)

# Make predictions using the Lasso model
y_pred_lasso = lasso_model.predict(X_test)

# Evaluate the Lasso model
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = mse_lasso ** 0.5
r2_lasso = r2_score(y_test, y_pred_lasso)

print("Lasso Regression Model:")
print(f"Mean Squared Error (MSE): {mse_lasso}")
print(f"Root Mean Squared Error (RMSE): {rmse_lasso}")
print(f"R-squared (R2): {r2_lasso}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess your data (similar to your initial code)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Ridge and Lasso models with your chosen alpha values
ridge_alpha = 20.0  # Adjust the alpha as needed
ridge_model = Ridge(alpha=ridge_alpha)

lasso_alpha = 10.0  # Adjust the alpha as needed
lasso_model = Lasso(alpha=lasso_alpha)

# Train the Ridge and Lasso models
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

# Make predictions using both models
y_pred_ridge = ridge_model.predict(X_test)
y_pred_lasso = lasso_model.predict(X_test)

# Create DataFrames for actual prices and predictions
results_df = pd.DataFrame({'Actual': y_test, 'Ridge Predicted': y_pred_ridge, 'Lasso Predicted': y_pred_lasso})

# Print the first few rows of the DataFrame to compare actual and predicted prices
print(results_df.head())

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load your time series data
# Replace 'preprocessed_data.csv' with the path to your dataset
df = pd.read_csv('preprocessed_data.csv')

# Convert 'Price' column to numeric
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

# Create a combined categorical column for time using 'Year' and 'Month'
df['time'] = df['Year'].astype(str) + '-' + df['Month'].astype(str)

# Plot your time series data
plt.plot(df['Price'])
plt.title('Price Over Time')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

# Check for stationarity, and apply differencing if needed
# Example:
# from statsmodels.tsa.stattools import adfuller
# result = adfuller(df['Price'])
# print('ADF Statistic:', result[0])
# print('p-value:', result[1])

# Differencing if needed
# df_diff = df['Price'].diff().dropna()

# Plot ACF and PACF to determine the order of ARIMA(p, d, q)
# plot_acf(df_diff)
# plot_pacf(df_diff)
# plt.show()

# Fit ARIMA model
# Example: ARIMA(1, 1, 1)
model = ARIMA(df['Price'], order=(2, 0, 2))
result = model.fit()

# Get forecast
n_forecast = 10  # Adjust as needed
forecast = result.get_forecast(steps=n_forecast)

# Extract the forecasted values and confidence intervals
forecast_values = forecast.predicted_mean
conf_int = forecast.conf_int()

# Plot the predicted values along with confidence interval
plt.plot(df['Price'], label='Actual')
plt.plot(forecast_values.index, forecast_values, color='red', label='Forecast')
plt.fill_between(forecast_values.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='red', alpha=0.1)
plt.title('ARIMA Forecast')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Evaluate the model performance
actual_values = df['Price']
predicted_values = result.fittedvalues  # or result.predict(start=..., end=...)
mse = mean_squared_error(actual_values, predicted_values)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_values, predicted_values)
r2 = r2_score(actual_values, predicted_values)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

n_forecast = 10
future_forecasts = result.predict(start=len(df), end=len(df) + n_forecast - 1, typ='levels')
future_forecasts.reset_index(drop=True, inplace=True)

comparison_df = pd.DataFrame({'Actual Price': df['Price'].tail(n_forecast).values, 'Future Price': future_forecasts})
print(comparison_df)

# Example code to plot actual vs. predicted values
plt.plot(actual_values, label='Actual')
plt.plot(predicted_values, label='Predicted', color='red')
plt.legend()
plt.show()

best_mse = float('inf')
best_order = None

for p in range(3):
    for d in range(3):
        for q in range(3):
            model = ARIMA(df['Price'], order=(p, d, q))
            result = model.fit()
            predicted_values = result.fittedvalues
            mse = mean_squared_error(actual_values, predicted_values)

            if mse < best_mse:
                best_mse = mse
                best_order = (p, d, q)

print(f"Best ARIMA Order: {best_order}")

# Use the best order parameters obtained from the hyperparameter tuning
best_p, best_d, best_q = 2, 0, 2  # Change these values to the optimal ones

# Fit the ARIMA model with the best parameters
best_model = ARIMA(df['Price'], order=(best_p, best_d, best_q))
best_result = best_model.fit()
best_predicted_values = best_result.fittedvalues

# Evaluate the performance of the best model
best_mse = mean_squared_error(actual_values, best_predicted_values)
best_rmse = np.sqrt(best_mse)
best_mae = mean_absolute_error(actual_values, best_predicted_values)
best_r2 = r2_score(actual_values, best_predicted_values)

print(f"Best ARIMA Order: ({best_p}, {best_d}, {best_q})")
print(f"Best Model Mean Squared Error (MSE): {best_mse}")
print(f"Best Model Root Mean Squared Error (RMSE): {best_rmse}")
print(f"Best Model Mean Absolute Error (MAE): {best_mae}")
print(f"Best Model R-squared (R2): {best_r2}")

# Plot actual vs. predicted values
plt.plot(actual_values, label='Actual')
plt.plot(best_predicted_values, label='Best Predicted', color='red')
plt.legend()
plt.show()

# Random Forest on updated dataset (randomized rainfall, temperature and humidity data for 2020 and 2021)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('preprocessed_data_2.csv')

# Convert 'Price' column to numeric, converting non-numeric values to NaN
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Drop rows with NaN values in the 'Price' column
df.dropna(subset=['Price'], inplace=True)
df.dropna(inplace=True)

# Exclude 'Price' column from the features (X)
X = df.drop(columns=['Price'])

# Target variable (y) is 'Price'
y = df['Price']

# Convert categorical variables to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

actual_vs_predicted = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(actual_vs_predicted)

# Decision Tree on updated dataset (randomized rainfall, temperature and humidity data for 2020 and 2021)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('preprocessed_data.csv')

# Split the data into training and testing sets (if not already done)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Regressor model
dt_model = DecisionTreeRegressor(random_state=42)

# Train the Decision Tree model on the training data
dt_model.fit(X_train, y_train)

# Make predictions using the Decision Tree model
y_pred_dt = dt_model.predict(X_test)

# Evaluate the Decision Tree model
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = mse_dt**0.5
r2_dt = r2_score(y_test, y_pred_dt)

print("Decision Tree Model:")
print(f"Mean Squared Error (MSE): {mse_dt}")
print(f"Root Mean Squared Error (RMSE): {rmse_dt}")
print(f"R-squared (R2): {r2_dt}")

actual_vs_predicted = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(actual_vs_predicted)

"""Sarima"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df

df['Month'] = df['Month'].apply(lambda x: (x + 1))

df



df

df.dropna(axis=0, inplace=True)

unique_values = df['Month'].unique()
df.dropna(inplace=True)
print(unique_values)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

df['time'] = df['Year'].astype(str) + '-' + df['Month'].astype(str)

# Plot your time series data
plt.plot(df['Price'])
plt.title('Price Over Time')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

df

unique_values = df['Year'].unique()
print(unique_values)

df['Year'] = df['Year'].replace(0, 2022)
df['Year'] = df['Year'].replace(1, 2022)

df

import pandas as pd

# Sample DataFrame with 'year' and 'month' columns
data = {'year': [2023, 2022, 2021], 'month': [6, 5, 11]}
df1 = pd.DataFrame(data)

# Create a new 'Date' column by combining 'year' and 'month'
df1['Date'] = pd.to_datetime(df1['year'].astype(str) + '-' + df1['month'].apply(lambda x: str(x).zfill(2)) + '-01')

# Print the DataFrame to see the changes
print(df1)

df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')

df

df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))
df.set_index('Date', inplace=True)

# Define SARIMA model
order = (2, 0, 2)  # Non-seasonal order (p, d, q)
seasonal_order = (1, 1, 1, 12)  # Seasonal order (P, D, Q, s)

# Create and fit the SARIMA model
model = SARIMAX(df['Price'], order=order, seasonal_order=seasonal_order, exog=None)
result = model.fit()

# Get forecast
n_forecast = 10  # Adjust as needed
forecast = result.get_forecast(steps=n_forecast)

# Extract the forecasted values and confidence intervals
forecast_values = forecast.predicted_mean
conf_int = forecast.conf_int()

# Plot the predicted values along with confidence interval
plt.plot(df['Price'], label='Actual')
plt.plot(forecast_values.index, forecast_values, color='red', label='Forecast')
plt.fill_between(forecast_values.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='red', alpha=0.1)
plt.title('SARIMA Forecast')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Evaluate the model performance
actual_values = df['Price']
predicted_values = result.fittedvalues
mse = mean_squared_error(actual_values, predicted_values)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_values, predicted_values)
r2 = r2_score(actual_values, predicted_values)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# Get future forecasts
n_forecast = 10
future_forecasts = result.forecast(steps=n_forecast)

comparison_df = pd.DataFrame({'Actual Price': df['Price'].tail(n_forecast).values, 'Future Price': future_forecasts.values})
print(comparison_df)