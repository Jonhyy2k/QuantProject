#do the code but ask for ticker and have same thinking method
import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
import websocket
import time
from threading import Thread
from collections import deque
import random
import traceback

# XTB API credentials (use your demo account email and password)
# It's recommended to store these in environment variables or a config file
XTB_USER_ID = "50540163"  # Replace with your XTB email
XTB_PASSWORD = "Jphost2005"  # Replace with your XTB password
XTB_WS_URL = "wss://ws.xtb.com/real"  # Demo server; use "real" for live accounts


# WebSocket connection manager with improved reconnection and heartbeat
class XTBClient:
    def __init__(self):
        self.ws = None
        self.logged_in = False
        self.response_data = {}
        self.last_command = None
        self.reconnect_count = 0
        self.max_reconnects = 5
        self.running = True
        self.heartbeat_thread = None

    def on_open(self, ws):
        print("[INFO] WebSocket connection opened.")
        self.reconnect_count = 0  # Reset reconnect counter on successful connection
        self.login()

    def on_message(self, ws, message):
        try:
            data = json.loads(message)

            # Handle login response
            if "streamSessionId" in data:
                self.logged_in = True
                print("[INFO] Logged in successfully.")
                # Start heartbeat after successful login
                if not self.heartbeat_thread or not self.heartbeat_thread.is_alive():
                    self.start_heartbeat()

            # This is how XTB API returns command responses - it has both status and returnData
            elif "status" in data and "returnData" in data:
                # Store the most recent command response
                # Since XTB doesn't include the command name in the response, use the most recent command
                if hasattr(self, 'last_command') and self.last_command:
                    self.response_data[self.last_command] = data["returnData"]
                    print(f"[DEBUG] Stored response for command: {self.last_command}")
                    self.last_command = None  # Reset last command
                else:
                    print(f"[DEBUG] Received response but can't match to command: {message[:100]}...")

            # Handle errors
            elif "errorDescr" in data:
                print(f"[ERROR] API error: {data['errorDescr']}")
                if hasattr(self, 'last_command') and self.last_command:
                    self.response_data[self.last_command] = {"error": data["errorDescr"]}
                    self.last_command = None

            else:
                print(f"[DEBUG] Received unhandled message: {message[:100]}...")

        except Exception as e:
            print(f"[ERROR] Error processing message: {e}, Message: {message[:100]}")

    def on_error(self, ws, error):
        print(f"[ERROR] WebSocket error: {error}")
        print(f"[DEBUG] WebSocket state: logged_in={self.logged_in}")

    def on_close(self, ws, close_status_code=None, close_msg=None):
        print(f"[INFO] WebSocket connection closed. Status: {close_status_code}, Message: {close_msg}")
        self.logged_in = False

        # Attempt to reconnect if we're still running and haven't exceeded max reconnects
        if self.running and self.reconnect_count < self.max_reconnects:
            self.reconnect_count += 1
            backoff_time = min(30, 2 ** self.reconnect_count)  # Exponential backoff up to 30 seconds
            print(
                f"[INFO] Attempting to reconnect in {backoff_time} seconds... (Attempt {self.reconnect_count}/{self.max_reconnects})")
            time.sleep(backoff_time)
            self.connect()
        elif self.reconnect_count >= self.max_reconnects:
            print(f"[ERROR] Maximum reconnection attempts ({self.max_reconnects}) reached. Giving up.")

    def connect(self):
        """Establish WebSocket connection with better error handling"""
        try:
            if self.ws is not None:
                self.ws.close()

            self.ws = websocket.WebSocketApp(
                XTB_WS_URL,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )

            # Start WebSocket connection in a separate thread
            websocket_thread = Thread(target=self.ws.run_forever)
            websocket_thread.daemon = True  # Allow thread to exit when main program exits
            websocket_thread.start()

            # Wait for connection and login
            timeout = time.time() + 15  # 15s timeout
            while not self.logged_in and time.time() < timeout:
                time.sleep(0.5)

            if not self.logged_in:
                print("[WARNING] Connection established but login timed out")
                return False

            return True

        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            return False

    def start_heartbeat(self):
        """Start heartbeat thread to keep connection alive"""

        def heartbeat_worker():
            print("[INFO] Starting heartbeat service")
            heartbeat_interval = 30  # seconds
            while self.running and self.logged_in:
                try:
                    # Use a lightweight command as heartbeat
                    status_cmd = {
                        "command": "ping",
                        "arguments": {}
                    }
                    if self.ws and self.ws.sock and self.ws.sock.connected:
                        self.ws.send(json.dumps(status_cmd))
                        print("[DEBUG] Heartbeat sent")
                    else:
                        print("[WARNING] Cannot send heartbeat, connection not active")
                        break
                except Exception as e:
                    print(f"[ERROR] Heartbeat error: {e}")
                    break

                # Sleep for the heartbeat interval
                time.sleep(heartbeat_interval)

            print("[INFO] Heartbeat service stopped")

        # Start heartbeat in a separate thread
        self.heartbeat_thread = Thread(target=heartbeat_worker)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()

    def send_command(self, command, arguments=None):
        """Send command to XTB API with retry logic"""
        if not self.logged_in and command != "login":
            print("[ERROR] Not logged in yet.")
            return None

        max_retries = 3
        for attempt in range(max_retries):
            try:
                payload = {"command": command}
                if arguments:
                    payload["arguments"] = arguments

                # Store command in response_data and track the last command
                self.response_data[command] = None
                self.last_command = command

                # Convert to JSON and send
                payload_str = json.dumps(payload)
                print(f"[DEBUG] Sending: {payload_str[:100]}")

                if not self.ws or not self.ws.sock or not self.ws.sock.connected:
                    print("[ERROR] WebSocket not connected")
                    self.connect()  # Try to reconnect
                    if not self.logged_in:
                        return None

                self.ws.send(payload_str)

                # Wait for response with timeout - INCREASED FROM 10 TO 30 SECONDS
                timeout = time.time() + 30  # 30s timeout
                while self.response_data[command] is None and time.time() < timeout:
                    time.sleep(0.1)

                if self.response_data[command] is None:
                    print(
                        f"[WARNING] Timeout waiting for response to command: {command}, attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:  # Only wait if we're going to retry
                        time.sleep(2 * (attempt + 1))  # Exponential backoff
                        continue
                else:
                    return self.response_data.get(command)

            except Exception as e:
                print(f"[ERROR] Error sending command {command}: {e}")
                if attempt < max_retries - 1:  # Only wait if we're going to retry
                    time.sleep(2 * (attempt + 1))

        return None  # Return None if all retries failed

    def login(self):
        """Log in to XTB API"""
        login_cmd = {
            "command": "login",
            "arguments": {"userId": XTB_USER_ID, "password": XTB_PASSWORD}
        }
        print("[DEBUG] Sending login command")
        self.last_command = "login"  # Set this for the login command too

        try:
            self.ws.send(json.dumps(login_cmd))
        except Exception as e:
            print(f"[ERROR] Failed to send login command: {e}")

    def disconnect(self):
        """Cleanly disconnect from XTB"""
        self.running = False  # Stop reconnection attempts and heartbeat

        if self.logged_in:
            try:
                # Try to logout properly
                logout_cmd = {"command": "logout"}
                self.ws.send(json.dumps(logout_cmd))
                time.sleep(1)  # Give it a moment to process
            except:
                pass  # Ignore errors during logout

        if self.ws:
            try:
                self.ws.close()
            except:
                pass

        print("[INFO] Disconnected from XTB")


# Function to fetch all listed stocks
def get_all_stocks(client):
    print("[INFO] Fetching all listed stocks from XTB...")

    # Send command with empty arguments object
    response = client.send_command("getAllSymbols", {})

    if response is None:
        print("[ERROR] Failed to fetch stock list.")
        return []

    print(f"[DEBUG] Response type: {type(response)}, Sample: {str(response)[:100]}")

    try:
        if isinstance(response, list):
            symbols = [item["symbol"] for item in response if isinstance(item, dict) and "symbol" in item]
            print(f"[INFO] Total stocks retrieved: {len(symbols)}")
            return symbols
        else:
            print(f"[ERROR] Unexpected response format: {str(response)[:100]}")
            return []
    except Exception as e:
        print(f"[ERROR] Error processing stock list: {e}")
        traceback.print_exc()  # Added traceback
        return []


# Function to fetch historical stock data
def get_stock_data(client, symbol):
    print(f"[INFO] Fetching historical data for: {symbol}")
    # XTB uses UNIX timestamps in milliseconds (last 1 year)
    end_time = int(time.time() * 1000)
    start_time = end_time - (365 * 24 * 60 * 60 * 1000)  # 1 year ago
    arguments = {
        "info": {
            "symbol": symbol,
            "period": 1440,  # Daily (1440 minutes)
            "start": start_time,
            "end": end_time
        }
    }
    response = client.send_command("getChartLastRequest", arguments)
    if response is None or "rateInfos" not in response:
        print(f"[WARNING] No historical data for {symbol}. Response: {response}")
        return None
    try:
        df = pd.DataFrame(response["rateInfos"])
        df["time"] = pd.to_datetime(df["ctm"], unit="ms")
        df["close"] = df["close"] + df["open"]  # XTB gives delta, we want absolute close
        df = df.set_index("time")

        # Add more price data columns
        if "open" in df.columns and "close" in df.columns:
            df["4. close"] = df["close"]
            df["high"] = df["open"] + df["high"]  # XTB gives deltas
            df["low"] = df["open"] + df["low"]  # XTB gives deltas
            df["volume"] = df["vol"]
            print(f"[DEBUG] Processed data for {symbol}: {len(df)} records")  # Added debug
            return df[["open", "high", "low", "4. close", "volume"]]
        else:
            print(f"[WARNING] Missing required columns in {symbol} data")
            return None
    except Exception as e:
        print(f"[ERROR] Error processing data for {symbol}: {e}")
        traceback.print_exc()  # Added traceback
        return None


# Calculate technical indicators for Sigma
def calculate_technical_indicators(data):
    try:
        print(f"[DEBUG] Calculating technical indicators on data with shape: {data.shape}")  # Added debug
        df = data.copy()

        # Calculate returns
        df['returns'] = df['4. close'].pct_change().fillna(0)

        # Calculate volatility (20-day rolling standard deviation)
        df['volatility'] = df['returns'].rolling(window=20).std().fillna(0)

        # Calculate Simple Moving Averages
        df['SMA20'] = df['4. close'].rolling(window=20).mean().fillna(0)
        df['SMA50'] = df['4. close'].rolling(window=50).mean().fillna(0)

        # Calculate Relative Strength Index (RSI)
        delta = df['4. close'].diff().fillna(0)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().fillna(0)
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().fillna(0)
        rs = gain / loss.replace(0, np.nan)  # Replace 0 with NaN to avoid division by zero
        df['RSI'] = 100 - (100 / (1 + rs)).fillna(50)  # Default to neutral RSI

        # Calculate Bollinger Bands
        df['BB_middle'] = df['SMA20']
        df['BB_std'] = df['4. close'].rolling(window=20).std().fillna(0)
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)

        # Calculate MACD
        df['EMA12'] = df['4. close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['4. close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Calculate trading volume changes
        df['volume_change'] = df['volume'].pct_change().fillna(0)

        print(f"[DEBUG] Technical indicators calculated successfully. New shape: {df.shape}")  # Added debug
        return df
    except Exception as e:
        print(f"[ERROR] Error calculating technical indicators: {e}")
        traceback.print_exc()  # Added traceback
        return None

# PCA function to reduce dimensionality of features
def apply_pca(features_df):
    try:
        # Debug info about input
        print(f"[DEBUG] PCA input shape: {features_df.shape}")

        # Check if we have enough data
        if features_df.shape[0] < 5 or features_df.shape[1] < 5:
            print("[WARNING] Not enough data for PCA analysis")
            return None, None

        # Select numerical columns that aren't NaN
        numeric_df = features_df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.fillna(0)  # Fill NaN values

        print(f"[DEBUG] PCA numeric data shape: {numeric_df.shape}")

        # Standardize the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        # Apply PCA
        pca = PCA(n_components=5)  # Reduce to 5 principal components
        pca_results = pca.fit_transform(scaled_data)

        # Create a DataFrame with PCA results
        pca_df = pd.DataFrame(
            pca_results,
            columns=[f'PC{i + 1}' for i in range(pca_results.shape[1])],
            index=features_df.index
        )

        # Calculate explained variance for each component
        explained_variance = pca.explained_variance_ratio_

        print(f"[INFO] PCA explained variance: {explained_variance}")
        return pca_df, explained_variance
    except Exception as e:
        print(f"[ERROR] PCA failed: {e}")
        traceback.print_exc()  # Added traceback
        return None, None


# Prepare data for LSTM prediction
def prepare_lstm_data(data, time_steps=60):
    try:
        # Get closing prices
        prices = data['4. close'].values

        # Normalize the data
        scaler = StandardScaler()
        scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))

        # Create sequences
        X, y = [], []
        for i in range(len(scaled_prices) - time_steps):
            X.append(scaled_prices[i:i + time_steps])
            y.append(scaled_prices[i + time_steps])

        return np.array(X), np.array(y), scaler
    except Exception as e:
        print(f"[ERROR] Error preparing LSTM data: {e}")
        return None, None, None


# LSTM model for volatility prediction
def build_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(50)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


# Train LSTM model and get volatility prediction
def predict_with_lstm(data):
    try:
        if len(data) < 70:  # Need enough data for training
            print("[WARNING] Not enough data for LSTM model")
            return 0

        # Prepare data
        X, y, scaler = prepare_lstm_data(data)
        if X is None:
            return 0

        # Build and train model
        model = build_lstm_model((X.shape[1], 1))
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        # Make prediction for future volatility
        last_sequence = X[-1].reshape(1, X.shape[1], 1)
        predicted_scaled = model.predict(last_sequence, verbose=0)

        # Compare prediction with last actual value to estimate volatility change
        predicted_volatility_change = abs((predicted_scaled[0][0] - y[-1][0]) / y[-1][0])

        return predicted_volatility_change
    except Exception as e:
        print(f"[ERROR] Error in LSTM prediction: {e}")
        return 0


# DQN agent for reinforcement learning
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Dense(128, activation="relu", input_shape=(self.state_size,)),
            Dense(64, activation="relu"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Function to get DQN recommendation
def get_dqn_recommendation(data):
    try:
        if len(data) < 100:  # Need enough data
            print("[WARNING] Not enough data for DQN")
            return 0.5  # Neutral score

        # Prepare state features (last 10 days of returns and volatility)
        returns = data['returns'].values[-10:]
        volatility = data['volatility'].values[-10:]
        state = np.concatenate((returns, volatility))

        # Define action space: 0=Sell, 1=Hold, 2=Buy
        action_size = 3
        agent = DQNAgent(state_size=len(state), action_size=action_size)

        # Train agent with historical data
        for i in range(len(data) - 11):
            curr_returns = data['returns'].values[i:i + 10]
            curr_volatility = data['volatility'].values[i:i + 10]
            current_state = np.concatenate((curr_returns, curr_volatility)).reshape(1, 20)

            next_returns = data['returns'].values[i + 1:i + 11]
            next_volatility = data['volatility'].values[i + 1:i + 11]
            next_state = np.concatenate((next_returns, next_volatility)).reshape(1, 20)

            # Calculate reward based on next day's return
            reward = data['returns'].values[i + 10]

            # Take action based on current state
            action = agent.act(current_state)

            # Record experience
            agent.remember(current_state, action, reward, next_state, False)

            # Train on batch of experiences
            if len(agent.memory) > 32:
                agent.replay(32)

        # Get recommendation from trained agent
        state = state.reshape(1, len(state))
        action = agent.act(state)

        # Convert action to score: 0=0.0, 1=0.5, 2=1.0
        dqn_score = action / 2
        return dqn_score
    except Exception as e:
        print(f"[ERROR] Error in DQN recommendation: {e}")
        return 0.5  # Neutral score


# Calculate Sigma metric combining all analysis tools
def calculate_sigma(data):
    try:
        # 1. Calculate technical indicators
        indicators_df = calculate_technical_indicators(data)
        if indicators_df is None:
            return None

        # 2. Apply PCA to reduce feature dimensionality
        pca_results, pca_variance = apply_pca(indicators_df.iloc[-30:])  # Use last 30 days
        if pca_results is None:
            return None

        # 3. Get LSTM volatility prediction
        lstm_prediction = predict_with_lstm(data)

        # 4. Get DQN recommendation
        dqn_recommendation = get_dqn_recommendation(indicators_df)

        # 5. Calculate traditional volatility (for comparison)
        traditional_volatility = indicators_df['volatility'].iloc[-1]

        # Get more technical signals
        latest = indicators_df.iloc[-1]

        # RSI signal (0-1 scale, higher means oversold/bullish)
        rsi_signal = (max(0, min(100, latest['RSI'])) - 30) / 70

        # MACD signal (-1 to 1 scale)
        macd_signal = np.tanh(latest['MACD'] * 10)

        # SMA trend signal (-1 to 1 scale)
        sma_signal = np.tanh((latest['SMA20'] / latest['SMA50'] - 1) * 10)

        # Bollinger Band position (0-1 scale)
        bb_range = latest['BB_upper'] - latest['BB_lower']
        if bb_range > 0:
            bb_position = (latest['4. close'] - latest['BB_lower']) / bb_range
        else:
            bb_position = 0.5

        # Volume signal (0-1 scale)
        volume_signal = min(1, max(0, (latest['volume_change'] + 0.1) / 0.2))

        # Calculate final Sigma score (0-1 scale)
        # Combine all signals with weights
        sigma = (
                0.20 * traditional_volatility * 25 +  # Traditional volatility (scaled)
                0.15 * lstm_prediction * 10 +  # LSTM prediction
                0.15 * dqn_recommendation +  # DQN recommendation
                0.15 * rsi_signal +  # RSI signal
                0.15 * (macd_signal + 1) / 2 +  # MACD signal (converted to 0-1)
                0.10 * (sma_signal + 1) / 2 +  # SMA trend (converted to 0-1)
                0.05 * bb_position +  # Bollinger Band position
                0.05 * volume_signal  # Volume signal
        )

        # Ensure sigma is between 0 and 1
        sigma = max(0, min(1, sigma))

        # For detailed analysis, return a dictionary
        analysis_details = {
            "sigma": sigma,
            "traditional_volatility": traditional_volatility,
            "lstm_prediction": lstm_prediction,
            "dqn_recommendation": dqn_recommendation,
            "rsi_signal": latest['RSI'],
            "macd": latest['MACD'],
            "sma_trend": latest['SMA20'] / latest['SMA50'] - 1,
            "bb_position": bb_position,
            "volume_change": latest['volume_change'],
            "pca_variance": pca_variance,
            "last_price": latest['4. close']
        }

        return analysis_details
    except Exception as e:
        print(f"[ERROR] Error calculating Sigma: {e}")
        return None


# Function to get recommendation based on Sigma score
def get_sigma_recommendation(sigma):
    if sigma > 0.8:
        return "STRONG BUY - High Sigma indicates significant trading opportunity with multiple confirming signals"
    elif sigma > 0.6:
        return "BUY - Good Sigma with positive technical indicators"
    elif sigma > 0.4:
        return "HOLD - Moderate Sigma with mixed signals"
    elif sigma > 0.2:
        return "SELL - Low Sigma suggests limited profit potential"
    else:
        return "STRONG SELL - Very low Sigma indicates poor trading conditions"


# Main function to analyze stocks with improved error handling
def analyze_stocks():
    print("[INFO] Starting advanced stock analysis with Sigma metric...")
    client = XTBClient()

    # Connect with retry logic
    connection_attempts = 0
    max_connection_attempts = 3

    while connection_attempts < max_connection_attempts:
        print(f"[INFO] Connection attempt {connection_attempts + 1}/{max_connection_attempts}")
        client.connect()

        # Wait for login with timeout
        login_timeout = time.time() + 15
        while not client.logged_in and time.time() < login_timeout:
            time.sleep(1)

        if client.logged_in:
            break

        connection_attempts += 1
        if connection_attempts < max_connection_attempts:
            print(f"[WARNING] Connection failed, retrying in {connection_attempts * 3} seconds...")
            time.sleep(connection_attempts * 3)  # Exponential backoff

    if not client.logged_in:
        print("[ERROR] Failed to connect to XTB after multiple attempts. Exiting.")
        return []

    # Get all available stocks
    stock_list = get_all_stocks(client)
    print(f"[INFO] Total stocks retrieved: {len(stock_list)}")
    if not stock_list:
        print("[ERROR] No stocks found. Exiting analysis.")
        client.disconnect()
        return []

    selected_stocks = []
    unique_stocks = set()  # Track unique stock symbols
    success_count = 0
    fail_count = 0

    try:
        for stock in stock_list[:20]:  # Process first 20 stocks for testing
            if stock in unique_stocks:  # Skip if stock already processed
                print(f"[INFO] Skipping duplicate stock: {stock}")
                continue

            # Check if still connected
            if not client.logged_in:
                print("[WARNING] Connection lost during analysis. Attempting to reconnect...")
                client.connect()
                if not client.logged_in:
                    print("[ERROR] Reconnection failed. Stopping analysis.")
                    break

            data = get_stock_data(client, stock)
            if data is not None and len(data) >= 100:  # Need enough historical data
                try:
                    # Calculate Sigma score combining all analysis tools
                    analysis = calculate_sigma(data)

                    if analysis:
                        sigma = analysis["sigma"]
                        recommendation = get_sigma_recommendation(sigma)

                        print(f"[INFO] {stock}: Sigma = {sigma:.5f}, Recommendation = {recommendation}")
                        print(f"[DEBUG] Analysis details: {analysis}")

                        # Keep stocks with meaningful sigma
                        if sigma > 0.1:
                            selected_stocks.append((stock, sigma, recommendation, analysis))
                            unique_stocks.add(stock)  # Add stock to the set of unique stocks
                        success_count += 1
                    else:
                        print(f"[WARNING] Could not calculate Sigma for {stock}")
                except Exception as e:
                    print(f"[ERROR] Error analyzing {stock}: {e}")
                    fail_count += 1
            else:
                fail_count += 1
                print(f"[WARNING] Insufficient historical data for {stock}.")

            time.sleep(1)  # Add a delay to avoid rate limits

    except KeyboardInterrupt:
        print("[INFO] Analysis interrupted by user.")
    except Exception as e:
        print(f"[ERROR] Unexpected error during analysis: {e}")
    finally:
        # Clean up and disconnect
        print("[INFO] Cleaning up connections...")
        client.disconnect()

    print(f"[INFO] Successfully analyzed {success_count} stocks.")
    print(f"[INFO] Failed to analyze {fail_count} stocks.")

    selected_stocks.sort(key=lambda x: x[1], reverse=True)
    if not selected_stocks:
        print("[WARNING] No stocks with significant Sigma found.")
    else:
        print(f"[INFO] Top {len(selected_stocks)} stocks by Sigma selected.")
    return selected_stocks


# Save stock list to text file
# Save stock list to text file
def save_to_text_file(stocks):
    file_path = os.path.expanduser("~/Desktop/stock_list.txt")
    if not stocks:
        print("[WARNING] No stocks to save.")
        return
    try:
        # Open the file in append mode
        with open(file_path, "a") as file:
            file.write("\n===== SIGMA-BASED TRADING RECOMMENDATIONS =====\n")
            file.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            file.write("SYMBOL - PRICE - SIGMA - RECOMMENDATION\n")
            file.write("--------------------------------------\n\n")

            for stock, sigma, rec, details in stocks:
                # Get the stock price from the details (assuming the last closing price)
                price = details.get("last_price", "N/A")  # Using get() with default value
                file.write(f"{stock} - Price: ${price:.2f} - Sigma: {sigma:.5f} - {rec}\n")
                file.write(f"  Details: RSI={details['rsi_signal']:.1f}, MACD={details['macd']:.4f}, SMA Trend={details['sma_trend']:.4f}\n\n")
    except Exception as e:
        print(f"[ERROR] Error saving to file: {e}")

# Execute the main function when the script is run
# At the bottom of your script, add this:
if __name__ == "__main__":
    print("[INFO] Starting stock analysis...")
    try:
        # First, try a simple test with one stock
        client = XTBClient()
        if client.connect():
            print("[TEST] Connection successful")
            # Get one stock
            stock = "AAPL.US_9"  # Test with Apple (adjust symbol as needed)
            data = get_stock_data(client, stock)
            if data is not None and len(data) > 0:
                print(f"[TEST] Got data for {stock}: {len(data)} records")
                # Try calculating indicators
                indicators = calculate_technical_indicators(data)
                if indicators is not None:
                    print("[TEST] Indicators calculated successfully")
                    # Try calculating sigma
                    analysis = calculate_sigma(data)
                    if analysis:
                        print(f"[TEST] Sigma calculation successful: {analysis['sigma']}")
                        print("[TEST] Basic testing passed, running full analysis...")
                        # Now run the full analysis
                        selected_stocks = analyze_stocks()
                        # Save results
                        save_to_text_file(selected_stocks)
                    else:
                        print("[ERROR] Sigma calculation failed during testing")
                else:
                    print("[ERROR] Indicator calculation failed during testing")
            else:
                print(f"[ERROR] Failed to get data for test stock {stock}")
            client.disconnect()
        else:
            print("[ERROR] Connection test failed")
    except Exception as e:
        print(f"[ERROR] Testing failed: {str(e)}")
        import traceback
        traceback.print_exc()