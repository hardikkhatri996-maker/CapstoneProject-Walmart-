#problem statement:
'''A retail store that has multiple outlets across the country are facing issues in managing the inventory - to match the demand with respect to supply.'''

'''
1. You are provided with the weekly sales data for their various outlets. Use statistical
analysis, EDA, outlier analysis, and handle the missing values to come up with various
insights that can give them a clear perspective on the following:
a. If the weekly sales are affected by the unemployment rate, if yes - which stores
are suffering the most?
b. If the weekly sales show a seasonal trend, when and what could be the reason?
c. Does temperature affect the weekly sales in any manner?
d. How is the Consumer Price index affecting the weekly sales of various stores?
e. Top performing stores according to the historical data.
f. The worst performing store, and how significant is the difference between the
highest and lowest performing stores.

2. Use predictive modeling techniques to forecast the sales for each store for the next 12
weeks.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("Walmart DataSet.csv")

print("Column names:", df.columns.tolist())

df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
print("Number of duplicate rows: ", df.duplicated().sum())

# data preprocessing
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Add holiday_flag column based on common US holidays
holidays = [
    '2010-11-26', '2010-12-31', '2011-11-25', '2011-12-30',  # Thanksgiving and Christmas
    '2012-11-23', '2012-12-28'  # Approximate dates
]
df['holiday_flag'] = df['Date'].dt.strftime('%Y-%m-%d').isin(holidays).astype(int)

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation matrix")
plt.show()

# Seasonal Trend Analysis
monthly_sales = df.groupby('Month')['Weekly_Sales'].mean()

plt.figure(figsize=(8,5))
monthly_sales.plot()
plt.title("Average Monthly Sales")
plt.ylabel("Sales")
plt.show()

# Holiday impact
holiday_sales = df.groupby('holiday_flag')['Weekly_Sales'].mean()
print("Average sales by holiday flag:")
print(holiday_sales)

# Store Performance Analysis
store_sales = df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False)

top_store = store_sales.idxmax()
worst_store = store_sales.idxmin()

print(f"Top performing store: {top_store} with total sales: ${store_sales[top_store]:,.2f}")
print(f"Worst performing store: {worst_store} with total sales: ${store_sales[worst_store]:,.2f}")
print(f"Difference between highest and lowest: ${(store_sales[top_store] - store_sales[worst_store]):,.2f}")

# Outliers Detection
plt.figure(figsize=(10,5))
sns.boxplot(x=df['Weekly_Sales'])
plt.title("Outlier Detection - Weekly_Sales")
plt.show()

# Statistical Analysis
print("\nCorrelation between Weekly Sales and Unemployment:")
print(df[['Weekly_Sales', 'Unemployment']].corr())

print("\nCorrelation between Weekly Sales and Temperature:")
print(df[['Weekly_Sales', 'Temperature']].corr())

print("\nCorrelation between Weekly Sales and CPI:")
print(df[['Weekly_Sales', 'CPI']].corr())

# Store-wise Impact Analysis

print("\nStore-wise Correlation (Sales vs Unemployment):")
store_unemp_corr = df.groupby('Store').apply(
    lambda x: x['Weekly_Sales'].corr(x['Unemployment'])
)
print(store_unemp_corr.sort_values())

print("\nStore-wise Correlation (Sales vs Temperature):")
store_temp_corr = df.groupby('Store').apply(
    lambda x: x['Weekly_Sales'].corr(x['Temperature'])
)
print(store_temp_corr.sort_values())

print("\nStore-wise Correlation (Sales vs CPI):")
store_cpi_corr = df.groupby('Store').apply(
    lambda x: x['Weekly_Sales'].corr(x['CPI'])
)
print(store_cpi_corr.sort_values())

print("\nStore most negatively affected by Unemployment:")
print(store_unemp_corr.idxmin(), "with correlation:", store_unemp_corr.min())

# Forecasting for ALL Stores


print("\nForecasting next 12 weeks for all stores...")

forecasts = {}

for store in df['Store'].unique():

    store_df = df[df['Store'] == store].sort_values('Date')
    store_df.set_index('Date', inplace=True)

    model = ExponentialSmoothing(
        store_df['Weekly_Sales'],
        seasonal_periods=52,
        trend='add',
        seasonal='add'
    )

    fitted_model = model.fit()
    forecast = fitted_model.forecast(12)

    forecasts[store] = forecast

print("Forecast completed for all stores.")

# Display forecast for first 3 stores as sample
for store in list(forecasts.keys())[:3]:
    print(f"\nStore {store} - Next 12 Weeks Forecast:")
    print(forecasts[store])

# Evaluate model for Top Performing Store
store_df = df[df['Store'] == top_store].sort_values('Date')
store_df.set_index('Date', inplace=True)
# Model Evaluation
train_size = int(len(store_df) * 0.8)
train = store_df[:train_size]
test = store_df[train_size:]

model_eval = ExponentialSmoothing(train['Weekly_Sales'],
                                   seasonal_periods=52,
                                   trend='add',
                                   seasonal='add')
fitted_eval = model_eval.fit()

predictions = fitted_eval.forecast(len(test))
mae = mean_absolute_error(test['Weekly_Sales'], predictions)
rmse = np.sqrt(mean_squared_error(test['Weekly_Sales'], predictions))

print(f"\nModel Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Visualization of forecast
sample_forecast = forecasts[top_store]

plt.plot(store_df.index, store_df['Weekly_Sales'], label='Actual Sales')
plt.plot(sample_forecast.index, sample_forecast, label='Forecast', color='red')

plt.title(f'Sales Forecast for Store {top_store}')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.legend()
plt.show()