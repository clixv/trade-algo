# Machine Learning-Based Trading Strategy (Corrected Version)

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, precision_recall_curve
import ta
import logging
import joblib

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ------------------------------
# Utility Functions
# ------------------------------

def save_model(model, threshold, filename='trading_model.joblib'):
    """Saves the model and threshold to a file."""
    joblib.dump({'model': model, 'threshold': threshold}, filename)
    logging.info(f"Model saved to {filename}")

def load_model(filename='trading_model.joblib'):
    """Loads the model and threshold from a file."""
    data = joblib.load(filename)
    logging.info(f"Model loaded from {filename}")
    return data['model'], data['threshold']

def download_data(ticker, start, end, timeout=60):
    df = yf.download(ticker, start=start, end=end, timeout=timeout, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}.")
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]
def backtest_placeholder(df_test, model, features, threshold, stop_loss=0.02, take_profit=0.04):
    """Placeholder function - the actual backtest function is defined later."""
    pass
# This is the NEW, corrected function
def add_features(df, feature_config):
    """Adds technical indicators to the dataframe."""
    # --- FIX: Use .squeeze() to ensure data is 1-dimensional ---
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()

    if 'rsi' in feature_config:
        df['rsi'] = ta.momentum.RSIIndicator(close=close, window=feature_config['rsi']['window']).rsi()
    if 'macd' in feature_config:
        df['macd'] = ta.trend.MACD(close=close).macd()
    if 'sma' in feature_config:
        for window in feature_config['sma']['windows']:
            df[f'sma_{window}'] = close.rolling(window=window).mean()
    if 'bollinger' in feature_config:
        bollinger = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        df['bb_pband'] = bollinger.bollinger_pband()
    if 'atr' in feature_config:
        df['atr'] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    if 'stoch_k' in feature_config:
        df['stoch_k'] = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3).stoch()
    if 'cci' in feature_config:
        df['cci'] = ta.trend.CCIIndicator(high=high, low=low, close=close, window=20).cci()

    df['future_return'] = close.pct_change().shift(-1)
    df['target'] = (df['future_return'] > 0).astype(int)

    df.dropna(inplace=True)
    return df

def split_data(df, feature_cols):
    """Splits data into training and testing sets."""
    X = df[feature_cols]
    y = df['target']
    split_idx = int(len(df) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Return the complete DataFrame for the test period
    df_test = df.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test, df_test

def hyperparameter_tuning(X_train, y_train):
    """Tunes hyperparameters for the model."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(
        estimator=rf, param_distributions=param_grid, n_iter=15,
        cv=tscv, scoring='f1', n_jobs=-1, random_state=42, verbose=0
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def optimize_threshold(model, X, y):
    """Finds the best probability threshold for classification."""
    probs = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
    f1_scores = np.nan_to_num(f1_scores)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

def backtest(df_test, model, features, threshold, stop_loss=0.02, take_profit=0.04):
    """
    Backtests the trading strategy with stop-loss, take-profit, and detailed metrics.
    """
    # --- STEP 1: Create a clean copy and reset the index to ensure 0-based integer positioning ---
    data = df_test.copy()
    data.reset_index(drop=True, inplace=True)

    # --- STEP 2: Generate predictions and signals on the clean DataFrame ---
    X_test_clean = data[features]
    probs = model.predict_proba(X_test_clean)[:, 1]
    data['proba'] = probs
    data['signal'] = (data['proba'] > threshold).astype(int)

    data['strategy_return'] = 0.0
    position = 0
    entry_price = 0

    # --- STEP 3: Loop using iloc for safer indexing ---
    signal_col = data.columns.get_loc('signal')
    open_col = data.columns.get_loc('Open')
    low_col = data.columns.get_loc('Low')
    high_col = data.columns.get_loc('High')
    strategy_return_col = data.columns.get_loc('strategy_return')
    
    for i in range(len(data) - 1):
        if position == 0 and data.iloc[i, signal_col].item() == 1:
            position = 1
            entry_price = float(data.iloc[i + 1, open_col])

        elif position == 1:
            low_price = float(data.iloc[i + 1, low_col])
            high_price = float(data.iloc[i + 1, high_col])

            # Check for stop-loss
            if low_price <= entry_price * (1 - stop_loss):
                exit_price = entry_price * (1 - stop_loss)
                data.iloc[i + 1, strategy_return_col] = (exit_price / entry_price) - 1
                position = 0
            # Check for take-profit
            elif high_price >= entry_price * (1 + take_profit):
                exit_price = entry_price * (1 + take_profit)
                data.iloc[i + 1, strategy_return_col] = (exit_price / entry_price) - 1
                position = 0

    # --- Calculation and Reporting ---
    data['log_future_return'] = np.log(1 + data['Close'].pct_change())
    data['log_strategy_return'] = np.log(1 + data['strategy_return'])
    data.fillna(0, inplace=True)

    cumulative_returns = pd.DataFrame({
        'Buy & Hold': data['log_future_return'].cumsum(),
        'Strategy': data['log_strategy_return'].cumsum()
    }).pipe(np.exp)

    cumulative_returns.plot(title='Cumulative Returns: Strategy vs. Buy & Hold', figsize=(12, 8), grid=True)
    plt.ylabel('Cumulative Return (1 = 100%)')
    plt.show()

    logging.info("--- Backtest Results ---")
    strategy_returns_trades = data['strategy_return'][data['strategy_return'] != 0]
    if not strategy_returns_trades.empty:
        sharpe_ratio = strategy_returns_trades.mean() / strategy_returns_trades.std() * np.sqrt(252)
        previous_peaks = cumulative_returns['Strategy'].cummax()
        drawdown = (cumulative_returns['Strategy'] - previous_peaks) / previous_peaks

        logging.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logging.info(f"Maximum Drawdown: {drawdown.min():.2%}")
        logging.info(f"Win Rate: {(strategy_returns_trades > 0).sum() / len(strategy_returns_trades) * 100:.2f}%")
        logging.info(f"Total Trades: {len(strategy_returns_trades)}")
    else:
        logging.info("No trades were made.")

def run_pipeline(ticker='AAPL', start='2023-01-01', end='2024-12-31'):
    logging.info(f"\n--- Running ML Trading Strategy for {ticker} ---")

    feature_conf = {
        'rsi': {'window': 14}, 'macd': {}, 'sma': {'windows': [10, 50]},
        'bollinger': {}, 'atr': {}, 'stoch_k': {}, 'cci': {}
    }

    df = download_data(ticker, start, end)
    df = add_features(df, feature_conf)

    features = [f for f in feature_conf.keys() if f != 'sma']
    if 'sma' in feature_conf:
        features.extend([f'sma_{w}' for w in feature_conf['sma']['windows']])
    features = [f for f in features if f in df.columns]

    X_train, X_test, y_train, y_test, df_test = split_data(df, features)

    logging.info("\n--- Tuning Hyperparameters ---")
    model, best_params = hyperparameter_tuning(X_train, y_train)
    logging.info(f"Best Parameters: {best_params}")

    logging.info("\n--- Evaluating Model ---")
    y_pred = model.predict(X_test)
    logging.info(classification_report(y_test, y_pred))

    logging.info("\n--- Optimizing Threshold ---")
    threshold = optimize_threshold(model, X_train, y_train)
    logging.info(f"Optimal Threshold: {threshold:.4f}")
    save_model(model, threshold, filename=f'{ticker}_model.joblib')

    logging.info("\n--- Backtesting Strategy ---")
    # This call now passes the list of feature names correctly
    backtest(df_test, model, features, threshold, stop_loss=0.02, take_profit=0.04)

# Run the pipeline
if __name__ == "__main__":
    run_pipeline()
