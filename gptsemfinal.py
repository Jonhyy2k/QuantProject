import os
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from alpha_vantage.timeseries import TimeSeries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Alpha Vantage API Key
API_KEY = "73KWO176IRABCOCJ"

# Function to get all listed stocks from a broker
# Alpha Vantage provides symbols via the "LISTING_STATUS" endpoint
BROKER_STOCKS_URL = f"https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={API_KEY}&datatype=csv"


# Function to fetch all listed stocks
def get_all_stocks():
    response = requests.get(BROKER_STOCKS_URL)
    if response.status_code == 200:
        data = pd.read_csv(pd.compat.StringIO(response.text))
        return data['symbol'].tolist()
    return []


# Function to fetch historical data
def get_stock_data(symbol):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if 'Time Series (Daily)' in data:
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        return df
    return None


# Apply PCA for factor modeling
def apply_pca(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    pca = PCA(n_components=5)
    return pca.fit_transform(scaled_data)


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
    selected_stocks = []

    for stock in stock_list[:50]:  # Limit for performance
        data = get_stock_data(stock)
        if data is not None:
            returns = data['4. close'].pct_change().dropna()
            volatility = returns.std()
            if volatility > 0.02:  # Selecting volatile stocks
                selected_stocks.append((stock, volatility))

    selected_stocks.sort(key=lambda x: x[1], reverse=True)
    return selected_stocks[:10]  # Top 10 most volatile stocks


# Save stock list to text file
def save_to_text_file(stocks):
    with open("stock_list.txt", "w") as file:
        for stock, vol in stocks:
            file.write(f"{stock} - Volatility: {vol:.5f}\n")


if __name__ == "__main__":
    selected_stocks = analyze_stocks()
    save_to_text_file(selected_stocks)
    print("Stock selection complete. Check stock_list.txt for details.")
