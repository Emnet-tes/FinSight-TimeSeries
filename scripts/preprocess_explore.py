import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# Asset tickers and date range
TICKERS = ['TSLA', 'BND', 'SPY']
START_DATE = '2015-07-01'
END_DATE = '2025-07-31'

# Fetch data
def fetch_data(ticker):
    df = yf.download(ticker, start=START_DATE, end=END_DATE)
    df['Ticker'] = ticker
    return df

data = pd.concat([fetch_data(t) for t in TICKERS])
data.reset_index(inplace=True)

# Data Cleaning
# Check for missing values
print('Missing values per column:')
print(data.isnull().sum())

# Fill missing values (forward fill, then backward fill)
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

# Ensure correct data types
data['Date'] = pd.to_datetime(data['Date'])
for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Basic statistics
print('Basic statistics:')
print(data.describe())

# EDA
for ticker in TICKERS:
    df = data[data['Ticker'] == ticker]
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'])
    plt.title(f'{ticker} Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.show()

    # Daily percentage change
    df['Daily Change'] = df['Close'].pct_change()
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Daily Change'])
    plt.title(f'{ticker} Daily Percentage Change')
    plt.xlabel('Date')
    plt.ylabel('Daily % Change')
    plt.show()

    # Rolling mean and std
    df['Rolling Mean'] = df['Close'].rolling(window=21).mean()
    df['Rolling Std'] = df['Close'].rolling(window=21).std()
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Close')
    plt.plot(df['Date'], df['Rolling Mean'], label='Rolling Mean (21 days)')
    plt.plot(df['Date'], df['Rolling Std'], label='Rolling Std (21 days)')
    plt.legend()
    plt.title(f'{ticker} Volatility Analysis')
    plt.show()

    # Outlier detection (z-score)
    df['Z-Score'] = (df['Daily Change'] - df['Daily Change'].mean()) / df['Daily Change'].std()
    outliers = df[np.abs(df['Z-Score']) > 3]
    print(f'Outliers for {ticker}:')
    print(outliers[['Date', 'Daily Change', 'Z-Score']])

    # Days with unusually high/low returns
    print(f'Days with high returns for {ticker}:')
    print(df[df['Daily Change'] > df['Daily Change'].quantile(0.99)][['Date', 'Daily Change']])
    print(f'Days with low returns for {ticker}:')
    print(df[df['Daily Change'] < df['Daily Change'].quantile(0.01)][['Date', 'Daily Change']])

    # Augmented Dickey-Fuller test
    adf_result = adfuller(df['Close'].dropna())
    print(f'ADF Statistic for {ticker}: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}')
    print('Critical Values:')
    for key, value in adf_result[4].items():
        print(f'   {key}: {value}')
    if adf_result[1] < 0.05:
        print('Series is stationary.')
    else:
        print('Series is NOT stationary. Consider differencing.')

    # Value at Risk (VaR)
    var_95 = np.percentile(df['Daily Change'].dropna(), 5)
    print(f'Value at Risk (95%) for {ticker}: {var_95}')

    # Sharpe Ratio
    sharpe_ratio = df['Daily Change'].mean() / df['Daily Change'].std() * np.sqrt(252)
    print(f'Sharpe Ratio for {ticker}: {sharpe_ratio}')

# Save cleaned data
cleaned_path = 'data/cleaned_financial_data.csv'
data.to_csv(cleaned_path, index=False)
print(f'Cleaned data saved to {cleaned_path}')
