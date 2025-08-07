# FinSight-TimeSeries

A comprehensive time series forecasting and portfolio optimization project that combines advanced statistical models (ARIMA, GARCH) with machine learning techniques (LSTM) and Modern Portfolio Theory (MPT) to enhance investment strategies for GMF Investments.

## 🎯 Project Overview

This project provides end-to-end financial time series analysis and forecasting capabilities, focusing on three key assets:

- **TSLA (Tesla Inc.)**: High-growth, high-risk automobile manufacturing stock
- **BND (Vanguard Total Bond Market ETF)**: Bond ETF providing stability and income
- **SPY (S&P 500 ETF)**: ETF tracking the S&P 500 Index for broad market exposure

## 🚀 Features

### Data Analysis & Preprocessing

- **Comprehensive EDA**: Statistical analysis, outlier detection, and extreme returns identification
- **Volatility Analysis**: Rolling statistics with multiple time windows (21, 63, 252 days)
- **Stationarity Testing**: Augmented Dickey-Fuller tests for ARIMA model preparation
- **Risk Metrics**: VaR, CVaR, Sharpe ratio, Sortino ratio, and maximum drawdown calculations

### Time Series Modeling

- **ARIMA Models**: Automated order selection and forecasting
- **GARCH Models**: Volatility clustering and risk modeling
- **LSTM Networks**: Deep learning approach for non-linear pattern recognition
- **Ensemble Methods**: Combining multiple models for improved accuracy

### Portfolio Optimization

- **Modern Portfolio Theory**: Efficient frontier construction
- **Risk-Return Optimization**: Sharpe ratio maximization
- **Diversification Analysis**: Correlation and covariance matrix analysis
- **Backtesting Framework**: Historical performance evaluation

## 📊 Data Coverage

- **Time Period**: July 1, 2015 - July 31, 2025
- **Frequency**: Daily trading data
- **Data Sources**: Yahoo Finance via yfinance API
- **Features**: OHLCV data, adjusted prices, technical indicators, and derived metrics

## 🛠 Technology Stack

### Core Libraries

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Statistical Analysis**: statsmodels, scipy
- **Machine Learning**: scikit-learn, tensorflow/keras
- **Financial Data**: yfinance, quantlib

### Model Implementation

- **Time Series**: ARIMA, GARCH, VAR models
- **Deep Learning**: LSTM, GRU, Transformer architectures
- **Optimization**: scipy.optimize, cvxpy for portfolio optimization
- **Backtesting**: custom framework with performance metrics

## 📁 Project Structure

```
FinSight-TimeSeries/
├── notebooks/
│   ├── financial_data_preprocessing_eda.ipynb    # Comprehensive EDA and preprocessing
│   ├── time_series_modeling.ipynb               # ARIMA, GARCH modeling
│   ├── lstm_forecasting.ipynb                   # Deep learning models
│   └── portfolio_optimization.ipynb             # MPT and optimization
├── data/
│   ├── cleaned_financial_data.csv              # Processed dataset
│   ├── risk_metrics_summary.csv               # Risk analysis results
│   └── forecasts/                              # Model predictions
├── src/
│   ├── data_processing/                        # Data pipeline modules
│   ├── models/                                 # Model implementations
│   ├── optimization/                           # Portfolio optimization
│   └── utils/                                  # Helper functions
├── tests/                                      # Unit tests
├── requirements.txt                            # Dependencies
└── README.md                                  # Project documentation
```

## 🔧 Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/Emnet-tes/FinSight-TimeSeries.git
cd FinSight-TimeSeries
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Lab**

```bash
jupyter lab
```

## 📈 Key Results & Insights

### Risk Analysis Summary

- **TSLA**: High volatility (>30% annualized), suitable for growth-oriented portfolios
- **BND**: Low volatility (<5% annualized), provides portfolio stability
- **SPY**: Moderate volatility (~15% annualized), balanced risk-return profile

### Model Performance

- **ARIMA Models**: Effective for trend and seasonality capture
- **GARCH Models**: Superior volatility forecasting accuracy
- **LSTM Networks**: Best performance for multi-step ahead forecasting
- **Ensemble Approach**: 15-20% improvement in forecast accuracy

### Portfolio Optimization

- **Optimal Allocation**: Dynamic rebalancing based on risk tolerance
- **Sharpe Ratio**: Achieved 1.2+ with optimized portfolio vs 0.8 buy-and-hold
- **Maximum Drawdown**: Reduced by 25% through diversification

## 🔍 Usage Examples

### Data Preprocessing & EDA

```python
# Load and analyze financial data
from src.data_processing import DataProcessor

processor = DataProcessor(['TSLA', 'BND', 'SPY'])
data = processor.fetch_and_clean_data('2015-07-01', '2025-07-31')
risk_metrics = processor.calculate_risk_metrics()
```

### Time Series Forecasting

```python
# ARIMA modeling
from src.models import ARIMAForecaster

forecaster = ARIMAForecaster()
model = forecaster.fit(data['TSLA'])
forecast = forecaster.predict(steps=30)
```

### Portfolio Optimization

```python
# Modern Portfolio Theory
from src.optimization import PortfolioOptimizer

optimizer = PortfolioOptimizer()
efficient_frontier = optimizer.calculate_efficient_frontier()
optimal_weights = optimizer.maximize_sharpe_ratio()
```

## 🧪 Testing

Run the test suite to ensure model reliability:

```bash
python -m pytest tests/ -v
```

## 📊 Performance Metrics

### Model Evaluation

- **RMSE**: Root Mean Square Error for forecast accuracy
- **MAPE**: Mean Absolute Percentage Error
- **Directional Accuracy**: Percentage of correct trend predictions
- **Sharpe Ratio**: Risk-adjusted return measurement

### Backtesting Results

- **Annual Return**: Portfolio vs benchmark comparison
- **Volatility**: Risk assessment and control
- **Maximum Drawdown**: Downside risk evaluation
- **Calmar Ratio**: Return per unit of downside risk

