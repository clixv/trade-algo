# Machine Learning-Based Trading Strategy (Modularized Version)

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, precision_recall_curve, f1_score
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

    df = yf.download(ticker, start=start, end=end, timeout=timeout)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}.")
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def add_features(df, feature_config):
    """
    Adds technical indicators to the dataframe based on a configuration.
    """
    close = df['Close']
    high = df['High']
    low = df['Low']

    # --- Feature Calculations ---
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
    # --- End of Feature Calculations ---

    df['future_return'] = close.pct_change().shift(-1)
    df['target'] = (df['future_return'] > 0).astype(int)

    df.dropna(inplace=True)
    return df

def split_data(df, feature_cols):
    X = df[feature_cols]
    y = df['target'] # <--- Correct use of single brackets
    split_idx = int(len(df) * 0.8)
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:], df.iloc[split_idx:].copy()

def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=15,
        cv=tscv,
        scoring='f1',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def optimize_threshold(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
    f1_scores = np.nan_to_num(f1_scores)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

def backtest(df_test, model, X_test, threshold, stop_loss=0.02, take_profit=0.04):
    """
    Backtests the trading strategy with stop-loss, take-profit, and detailed metrics.
    """
    probs = model.predict_proba(X_test)[:, 1]
    df_test['proba'] = probs
    df_test['signal'] = (df_test['proba'] > threshold).astype(int)
    
    df_test['strategy_return'] = 0.0
    position = 0
    entry_price = 0

    # Loop through the test data to simulate trades
    for i in range(len(df_test) - 1):
        if position == 0 and df_test['signal'].iloc[i] == 1:
            # Enter a new position on the next day's open
            position = 1
            entry_price = df_test['Open'].iloc[i + 1]
        
        elif position == 1:
            # Check for stop-loss or take-profit during the day
            if df_test['Low'].iloc[i + 1] <= entry_price * (1 - stop_loss):
                # Stop-loss triggered
                exit_price = entry_price * (1 - stop_loss)
                df_test.loc[df_test.index[i + 1], 'strategy_return'] = (exit_price / entry_price) - 1
                position = 0
            elif df_test['High'].iloc[i + 1] >= entry_price * (1 + take_profit):
                # Take-profit triggered
                exit_price = entry_price * (1 + take_profit)
                df_test.loc[df_test.index[i + 1], 'strategy_return'] = (exit_price / entry_price) - 1
                position = 0
            # Optional: Exit if signal turns to 0 (hold until next signal)
            # elif df_test['signal'].iloc[i] == 0:
            #     exit_price = df_test['Open'].iloc[i+1]
            #     df_test.loc[df_test.index[i+1], 'strategy_return'] = (exit_price / entry_price) - 1
            #     position = 0

    # --- Calculate Returns and Metrics ---
    
    # Calculate log returns for accurate compounding
    df_test['log_future_return'] = np.log(1 + df_test['Close'].pct_change())
    df_test['log_strategy_return'] = np.log(1 + df_test['strategy_return'])
    df_test.fillna(0, inplace=True) # Replace NaNs from log calculation

    cumulative_returns = pd.DataFrame({
        'Buy & Hold': df_test['log_future_return'].cumsum(),
        'Strategy': df_test['log_strategy_return'].cumsum()
    })
    cumulative_returns = np.exp(cumulative_returns)

    # --- Metrics Calculation ---
    cumulative_strategy_returns = cumulative_returns['Strategy']
    previous_peaks = cumulative_strategy_returns.cummax()
    drawdown = (cumulative_strategy_returns - previous_peaks) / previous_peaks
    max_drawdown = drawdown.min()

    strategy_returns_trades = df_test['strategy_return'][df_test['strategy_return'] != 0]
    winning_trades = (strategy_returns_trades > 0).sum()
    total_trades = len(strategy_returns_trades)
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    # --- Plotting ---
    cumulative_returns.plot(title=f'Cumulative Returns: Strategy vs. Buy & Hold for {X_test.index[0].year}-{X_test.index[-1].year}', figsize=(12, 8))
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (1 = 100%)')
    plt.grid(True)
    plt.show()

    # --- Final Report ---
    logging.info(f"--- Backtest Results ---")
    if not strategy_returns_trades.empty:
        sharpe_ratio = strategy_returns_trades.mean() / strategy_returns_trades.std() * np.sqrt(252)
        logging.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logging.info(f"Maximum Drawdown: {max_drawdown:.2%}")
        logging.info(f"Win Rate: {win_rate:.2f}%")
        logging.info(f"Total Trades: {total_trades}")
    else:
        logging.info("Sharpe Ratio: 0.00 (No trades were made)")
        
def run_pipeline(ticker='AAPL', start='2023-01-01', end='2024-12-31'):
    logging.info(f"\n--- Running ML Trading Strategy for {ticker} ---")
    df = download_data(ticker, start, end)
    # 1. Define the configuration for your features.
    feature_conf = {
        'rsi': {'window': 14},
        'macd': {},
        'sma': {'windows': [10, 50]},
        'bollinger': {},
        'atr': {},
        'stoch_k': {},
        'cci': {}
    }
    
    # 2. Pass the dictionary when you call add_features.
    # The original call was df = add_features(df). Update it to this:
    df = add_features(df, feature_conf)

    # 3. Dynamically create the list of feature names from the dictionary.
    # This replaces your old static list: features = ['rsi', 'macd', ...].
    features = []
    if 'rsi' in feature_conf: features.append('rsi')
    if 'macd' in feature_conf: features.append('macd')
    if 'sma' in feature_conf:
        for w in feature_conf['sma']['windows']: features.append(f'sma_{w}')
    if 'bollinger' in feature_conf: features.append('bb_pband')
    if 'atr' in feature_conf: features.append('atr')
    if 'stoch_k' in feature_conf: features.append('stoch_k')
    if 'cci' in feature_conf: features.append('cci')
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
    backtest(df_test, model, X_test, threshold, stop_loss=0.02, take_profit=0.04)
    

# Run
run_pipeline()
