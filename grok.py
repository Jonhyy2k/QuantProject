import os
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import time
import io  # For StringIO

# Alpha Vantage API Key
API_KEY = "73KWO176IRABCOCJ"


# Function to fetch all listed stocks
def get_all_stocks():
    url = f"https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={API_KEY}&datatype=csv"
    try:
        response = requests.get(url)
        response.raise_for_status()
        print("Raw response from LISTING_STATUS (first 500 chars):")
        print(response.text[:500])  # Debug: see what we’re getting
        data = pd.read_csv(io.StringIO(response.text))
        print("Columns in response:", data.columns.tolist())  # Debug: check columns
        if 'symbol' in data.columns:
            return data['symbol'].tolist()
        else:
            print("No 'symbol' column found in response.")
            return []
    except Exception as e:
        print(f"Error fetching stock list: {e}")
        return []


# Function to fetch historical data
def get_stock_data(symbol):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'Time Series (Daily)' in data:
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            return df
        else:
            print(f"No data for {symbol}")
            return None
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None


# Apply PCA for factor modeling
def apply_pca(data):
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        pca = PCA(n_components=5)
        return pca.fit_transform(scaled_data)
    except Exception as e:
        print(f"PCA error: {e}")
        return None


# Reinforcement Learning with Deep Q-Networks
def build_dqn_model(input_shape, action_space):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(action_space, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# LSTM for time series forecasting
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# Main function to analyze stocks
def analyze_stocks():
    stock_list = get_all_stocks()
    if not stock_list:
        print("No stocks retrieved. Exiting.")
        return []

    print(f"Retrieved {len(stock_list)} stocks. Processing first 50...")
    selected_stocks = []
    for i, stock in enumerate(stock_list[:50]):  # Limit to 50 as in original
        print(f"Processing {stock} ({i + 1}/50)")
        data = get_stock_data(stock)
        if data is not None and '4. close' in data.columns:
            returns = data['4. close'].pct_change().dropna()
            if len(returns) > 0:
                volatility = returns.std()
                if volatility > 0.02:  # Volatile stocks
                    selected_stocks.append((stock, volatility))
        time.sleep(12)  # Respect Alpha Vantage: 5 calls/min
    selected_stocks.sort(key=lambda x: x[1], reverse=True)
    return selected_stocks[:10]  # Top 10 volatile stocks


# Save stock list to text file
def save_to_text_file(stocks):
    try:
        with open("stock_list.txt", "w") as file:
            for stock, vol in stocks:
                file.write(f"{stock} - Volatility: {vol:.5f}\n")
    except Exception as e:
        print(f"Error writing to file: {e}")


if __name__ == "__main__":
    selected_stocks = analyze_stocks()
    if selected_stocks:
        save_to_text_file(selected_stocks)
        print("Stock selection complete. Check stock_list.txt for details.")
    else:
        print("No stocks selected.")