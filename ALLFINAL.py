import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import websocket
import time
from threading import Thread
from collections import deque
import random
import traceback
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# XTB API credentials (from environment variables)
XTB_USER_ID = os.environ.get("XTB_USER_ID", "50540163")  # Fallback to provided ID if env var not set
XTB_PASSWORD = os.environ.get("XTB_PASSWORD", "Jphost2005")  # Fallback to provided password if env var not set
XTB_WS_URL = os.environ.get("XTB_WS_URL", "wss://ws.xtb.com/real")  # Demo server; use "real" for live accounts

# Output file
OUTPUT_FILE = "XTB_STOCK_DATA_SET.txt"

# Global settings for batch processing
MAX_STOCKS_PER_BATCH = 100  # Limit number of stocks to analyze in a single run
BATCH_DELAY = 30  # Seconds to wait between batches to avoid API rate limits
MAX_EXECUTION_TIME_PER_STOCK = 180  # 3 minutes max per stock
MAX_TOTAL_RUNTIME = 240 * 3600  # 12 hours maximum total runtime


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
        self.command_lock = False  # Simple lock for commands

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
                    self.command_lock = False  # Release lock
                else:
                    print(f"[DEBUG] Received response but can't match to command: {message[:100]}...")

            # Handle errors
            elif "errorDescr" in data:
                print(f"[ERROR] API error: {data.get('errorDescr', 'Unknown error')}")
                if hasattr(self, 'last_command') and self.last_command:
                    self.response_data[self.last_command] = {"error": data.get("errorDescr", "Unknown error")}
                    self.last_command = None
                    self.command_lock = False  # Release lock

            else:
                print(f"[DEBUG] Received unhandled message: {message[:100]}...")

        except Exception as e:
            print(f"[ERROR] Error processing message: {e}, Message: {message[:100]}")
            self.command_lock = False  # Release lock in case of error

    def on_error(self, ws, error):
        print(f"[ERROR] WebSocket error: {error}")
        print(f"[DEBUG] WebSocket state: logged_in={self.logged_in}")
        self.command_lock = False  # Release lock in case of error

    def on_close(self, ws, close_status_code=None, close_msg=None):
        print(f"[INFO] WebSocket connection closed. Status: {close_status_code}, Message: {close_msg}")
        self.logged_in = False
        self.command_lock = False  # Release lock

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

        # Only start a new thread if one isn't already running
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            print("[INFO] Heartbeat thread already running")
            return

        self.heartbeat_thread = Thread(target=heartbeat_worker)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()

    def send_command(self, command, arguments=None):
        """Send command to XTB API with retry logic"""
        if not self.logged_in and command != "login":
            print("[ERROR] Not logged in yet.")
            return None

        # Check for command lock (simple concurrency control)
        timeout = time.time() + 5  # 5s timeout for lock
        while self.command_lock and time.time() < timeout:
            time.sleep(0.1)

        if self.command_lock:
            print(f"[ERROR] Command lock timeout for {command}")
            return None

        self.command_lock = True  # Acquire lock

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
                        self.command_lock = False  # Release lock
                        return None

                self.ws.send(payload_str)

                # Wait for response with timeout
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
                        self.command_lock = False  # Release lock
                else:
                    # Check for error in response
                    if isinstance(self.response_data[command], dict) and 'error' in self.response_data[command]:
                        print(f"[ERROR] API error for command {command}: {self.response_data[command]['error']}")
                        self.command_lock = False  # Release lock
                        return None

                    result = self.response_data.get(command)
                    self.command_lock = False  # Release lock
                    return result

            except Exception as e:
                print(f"[ERROR] Error sending command {command}: {e}")
                if attempt < max_retries - 1:  # Only wait if we're going to retry
                    time.sleep(2 * (attempt + 1))

        self.command_lock = False  # Release lock
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


# Function to get all available stock symbols from XTB
def get_all_stock_symbols(client):
    print("[INFO] Retrieving all available stock symbols from XTB API")

    try:
        response = client.send_command("getAllSymbols", {})

        if response is None:
            print("[ERROR] Failed to fetch stock list.")
            return []

        # Filter to get only valid stock symbols
        stocks = []

        if isinstance(response, list):
            for item in response:
                if isinstance(item, dict) and "symbol" in item:
                    # Extract symbol and additional info
                    symbol = item.get("symbol", "")
                    category = item.get("categoryName", "")
                    description = item.get("description", "")

                    # Make sure it's a stock - filter by category if needed
                    # This filtering criteria might need adjustment based on XTB's categories
                    if symbol and len(symbol) > 0:
                        stocks.append({"symbol": symbol,
                                       "category": category,
                                       "description": description})

        print(f"[INFO] Found {len(stocks)} total symbols")
        return stocks
    except Exception as e:
        print(f"[ERROR] Error getting stock symbols: {e}")
        traceback.print_exc()
        return []


# Function to fetch historical stock data
def get_stock_data(client, symbol):
    print(f"[INFO] Fetching historical data for: {symbol}")
    try:
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

        if response is None:
            print(f"[WARNING] No response from API for {symbol}")
            return None

        if "rateInfos" not in response or not response["rateInfos"]:
            print(f"[WARNING] No historical data for {symbol}. Response: {response}")
            return None

        df = pd.DataFrame(response["rateInfos"])

        if df.empty:
            print(f"[WARNING] Empty dataframe for {symbol}")
            return None

        df["time"] = pd.to_datetime(df["ctm"], unit="ms")
        df["close"] = df["close"] + df["open"]  # XTB gives delta, we want absolute close
        df = df.set_index("time")

        # Add more price data columns
        if "open" in df.columns and "close" in df.columns:
            df["4. close"] = df["close"]
            df["high"] = df["open"] + df["high"]  # XTB gives deltas
            df["low"] = df["open"] + df["low"]  # XTB gives deltas
            df["volume"] = df["vol"]

            # Check for NaN values
            for col in ["open", "high", "low", "4. close", "volume"]:
                if df[col].isna().any():
                    print(f"[WARNING] NaN values found in {col}, filling forward")
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

            print(f"[DEBUG] Processed data for {symbol}: {len(df)} records")
            return df[["open", "high", "low", "4. close", "volume"]]
        else:
            print(f"[WARNING] Missing required columns in {symbol} data")
            return None
    except Exception as e:
        print(f"[ERROR] Error processing data for {symbol}: {e}")
        traceback.print_exc()
        return None


# Calculate technical indicators for Sigma
def calculate_technical_indicators(data):
    try:
        print(f"[DEBUG] Calculating technical indicators on data with shape: {data.shape}")
        df = data.copy()

        # Check if data is sufficient
        if len(df) < 50:
            print("[WARNING] Not enough data for technical indicators calculation")
            return None

        # Calculate returns
        df['returns'] = df['4. close'].pct_change()
        df['returns'] = df['returns'].fillna(0)

        # Calculate volatility (20-day rolling standard deviation)
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volatility'] = df['volatility'].fillna(0)

        # Calculate Simple Moving Averages
        df['SMA20'] = df['4. close'].rolling(window=20).mean()
        df['SMA50'] = df['4. close'].rolling(window=50).mean()

        # Fill NaN values in SMAs with forward fill then backward fill
        df['SMA20'] = df['SMA20'].fillna(method='ffill').fillna(method='bfill')
        df['SMA50'] = df['SMA50'].fillna(method='ffill').fillna(method='bfill')

        # Calculate Relative Strength Index (RSI)
        delta = df['4. close'].diff()
        delta = delta.fillna(0)

        # Handle division by zero and NaN values in RSI calculation
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        # Handle zero avg_loss
        rs = np.zeros_like(avg_gain)
        valid_indices = avg_loss != 0
        rs[valid_indices] = avg_gain[valid_indices] / avg_loss[valid_indices]

        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)  # Default to neutral RSI (50)

        # Calculate Bollinger Bands
        df['BB_middle'] = df['SMA20']
        df['BB_std'] = df['4. close'].rolling(window=20).std()
        df['BB_std'] = df['BB_std'].fillna(0)
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)

        # Calculate MACD
        df['EMA12'] = df['4. close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['4. close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Calculate trading volume changes
        df['volume_change'] = df['volume'].pct_change()
        df['volume_change'] = df['volume_change'].fillna(0)

        # Final check for NaN values
        if df.isna().sum().sum() > 0:
            print(f"[WARNING] NaN values in technical indicators: {df.isna().sum()}")
            df = df.fillna(0)  # Fill any remaining NaNs with zeros

        print(f"[DEBUG] Technical indicators calculated successfully. New shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Error calculating technical indicators: {e}")
        traceback.print_exc()
        return None


# PCA function to reduce dimensionality of features
def apply_pca(features_df):
    try:
        # Debug info about input
        print(f"[DEBUG] PCA input shape: {features_df.shape}")

        # Check if we have enough data
        if features_df.shape[0] < 10 or features_df.shape[1] < 5:
            print(f"[WARNING] Not enough data for PCA analysis: {features_df.shape}")
            return None, None

        # Select numerical columns that aren't NaN
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude columns that are mostly NaN
        valid_cols = []
        for col in numeric_cols:
            if features_df[col].isna().sum() < len(features_df) * 0.5:  # At least 50% of values are not NaN
                valid_cols.append(col)

        if len(valid_cols) < 5:
            print(f"[WARNING] Not enough valid columns for PCA: {len(valid_cols)}")
            return None, None

        numeric_df = features_df[valid_cols].copy()

        # Fill remaining NaN values with column means
        for col in numeric_df.columns:
            if numeric_df[col].isna().any():
                numeric_df[col] = numeric_df[col].fillna(numeric_df[col].mean())

        print(f"[DEBUG] PCA numeric data shape after cleaning: {numeric_df.shape}")

        # Check for remaining NaN values
        if numeric_df.isna().sum().sum() > 0:
            print(f"[WARNING] NaN values still present after cleaning: {numeric_df.isna().sum().sum()}")
            # Replace remaining NaNs with 0
            numeric_df = numeric_df.fillna(0)

        # Standardize the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        # Apply PCA
        n_components = min(5, min(scaled_data.shape) - 1)
        pca = PCA(n_components=n_components)
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
        traceback.print_exc()
        return None, None


# Prepare data for LSTM prediction
def prepare_lstm_data(data, time_steps=30):  # Reduced from 60
    try:
        # Check if we have enough data
        if len(data) < time_steps + 10:
            print(f"[WARNING] Not enough data for LSTM: {len(data)} < {time_steps + 10}")
            return None, None, None

        # Get closing prices
        prices = data['4. close'].values

        # Check for NaN values
        if np.isnan(prices).any():
            print(f"[WARNING] NaN values in prices, filling with forward fill")
            # Fill NaN values using pandas Series methods
            prices = pd.Series(prices).fillna(method='ffill').fillna(method='bfill').values

        # Reshape to 2D for StandardScaler
        prices_2d = prices.reshape(-1, 1)

        # Normalize the data
        scaler = StandardScaler()
        scaled_prices = scaler.fit_transform(prices_2d)

        # Create sequences
        X, y = [], []
        for i in range(len(scaled_prices) - time_steps):
            X.append(scaled_prices[i:i + time_steps])
            y.append(scaled_prices[i + time_steps])

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Check shapes
        print(f"[DEBUG] LSTM data shapes: X={X.shape}, y={y.shape}")

        return X, y, scaler
    except Exception as e:
        print(f"[ERROR] Error preparing LSTM data: {e}")
        traceback.print_exc()
        return None, None, None


# Optimized LSTM model for volatility prediction
def build_lstm_model(input_shape):
    try:
        # Simpler model architecture
        inputs = Input(shape=input_shape)
        x = LSTM(30, return_sequences=False)(inputs)  # Single LSTM layer, fewer units
        outputs = Dense(1)(x)  # Direct output, no extra dense layers
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        return model
    except Exception as e:
        print(f"[ERROR] Error building LSTM model: {e}")
        traceback.print_exc()
        return None


# Train LSTM model and get volatility prediction with timeout
def predict_with_lstm(data):
    try:
        # Set a maximum execution time
        max_execution_time = 60  # 60 seconds max
        start_time = time.time()

        # Fail fast if not enough data
        if len(data) < 70:
            print("[WARNING] Not enough data for LSTM model")
            return 0

        # Use a smaller window for faster processing
        time_steps = 30  # Reduced from 60

        # Prepare data
        X, y, scaler = prepare_lstm_data(data, time_steps=time_steps)
        if X is None or y is None or scaler is None:
            print("[WARNING] Failed to prepare LSTM data")
            return 0

        # Check if we have enough data
        if len(X) < 10:
            print(f"[WARNING] Not enough data after preparation: {len(X)}")
            return 0

        # Build model
        model = build_lstm_model((X.shape[1], 1))
        if model is None:
            print("[WARNING] Failed to build LSTM model")
            return 0

        # Use a smaller subset of data for training if data is large
        max_samples = 200
        if len(X) > max_samples:
            # Use evenly spaced samples
            indices = np.linspace(0, len(X) - 1, max_samples, dtype=int)
            X_train = X[indices]
            y_train = y[indices]
        else:
            X_train = X
            y_train = y

        # Use try/except for model training
        try:
            # Check if we're still within time limit
            if time.time() - start_time > max_execution_time:
                print("[WARNING] LSTM execution time limit reached before training")
                # Use a very simplified prediction
                return data['volatility'].iloc[-5:].mean() / data['volatility'].iloc[-20:].mean()

            # Train model with fewer epochs and early stopping
            early_stop = EarlyStopping(monitor='loss', patience=1, verbose=0)

            # Use a timeout wrapper for model.fit
            fit_timeout = min(30, max_execution_time - (time.time() - start_time))

            # Set parameters to minimize training time
            model.fit(
                X_train, y_train,
                epochs=5,  # Reduced from 10
                batch_size=32,
                callbacks=[early_stop],
                verbose=0,
                shuffle=True
            )
        except Exception as e:
            print(f"[ERROR] LSTM model training failed: {e}")
            return 0

        # Make prediction for future volatility (if time permits)
        try:
            # Check time again
            if time.time() - start_time > max_execution_time:
                print("[WARNING] LSTM execution time limit reached before prediction")
                return 0.5  # Return a neutral value

            last_sequence = X[-1].reshape(1, X.shape[1], 1)
            predicted_scaled = model.predict(last_sequence, verbose=0)

            # Compare prediction with last actual value to estimate volatility change
            last_actual = y[-1][0]
            predicted_value = predicted_scaled[0][0]

            # Avoid division by zero
            if abs(last_actual) < 1e-6:
                predicted_volatility_change = abs(predicted_value)
            else:
                predicted_volatility_change = abs((predicted_value - last_actual) / last_actual)

            print(f"[DEBUG] LSTM prediction: {predicted_volatility_change}")
            return min(1.0, predicted_volatility_change)  # Cap at 1.0
        except Exception as e:
            print(f"[ERROR] LSTM prediction failed: {e}")
            return 0
    except Exception as e:
        print(f"[ERROR] Error in LSTM prediction: {e}")
        traceback.print_exc()
        return 0


# Optimized DQN Agent implementation
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)  # Reduced memory size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1  # Increased min epsilon for faster exploration/exploitation transition
        self.epsilon_decay = 0.9  # Faster decay
        self.model = self._build_model()
        self.max_training_time = 30  # Maximum training time in seconds

    def _build_model(self):
        try:
            # Simpler model architecture
            model = Sequential([
                Dense(64, activation="relu", input_shape=(self.state_size,)),  # Reduced from 128
                Dense(self.action_size, activation="linear")  # Removed middle layer
            ])
            model.compile(optimizer="adam", loss="mse")
            return model
        except Exception as e:
            print(f"[ERROR] Error building DQN model: {e}")
            # Even simpler fallback model
            try:
                model = Sequential([
                    Dense(32, activation="relu", input_shape=(self.state_size,)),
                    Dense(self.action_size, activation="linear")
                ])
                model.compile(optimizer="adam", loss="mse")
                return model
            except Exception as e2:
                print(f"[ERROR] Error building fallback DQN model: {e2}")
                return None

    def remember(self, state, action, reward, next_state, done):
        # Only add to memory if not full
        if len(self.memory) < self.memory.maxlen:
            self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        try:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            if self.model is None:
                return random.randrange(self.action_size)
            act_values = self.model.predict(state, verbose=0)
            return np.argmax(act_values[0])
        except Exception as e:
            print(f"[ERROR] Error in DQN act method: {e}")
            return random.randrange(self.action_size)

    def replay(self, batch_size):
        if len(self.memory) < batch_size or self.model is None:
            return

        # Add timeout mechanism
        start_time = time.time()

        try:
            # Limit batch size if memory is small
            actual_batch_size = min(batch_size, len(self.memory))
            minibatch = random.sample(self.memory, actual_batch_size)

            # Process in smaller chunks to prevent long-running fits
            chunk_size = 8  # Process 8 samples at a time
            for i in range(0, len(minibatch), chunk_size):
                chunk = minibatch[i:i + chunk_size]

                # Check timeout
                if time.time() - start_time > self.max_training_time:
                    print("[WARNING] DQN training timeout reached")
                    break

                # Process chunk
                states = np.vstack([x[0] for x in chunk])
                targets_f = self.model.predict(states, verbose=0)

                for j, (state, action, reward, next_state, done) in enumerate(chunk):
                    target = reward
                    if not done:
                        # Limit prediction to prevent hanging
                        try:
                            target = reward + self.gamma * np.amax(
                                self.model.predict(next_state, verbose=0)[0]
                            )
                        except:
                            # Fallback if prediction fails
                            target = reward

                    targets_f[j][action] = target

                # Fit with minimal epochs and tight timeout
                self.model.fit(
                    states,
                    targets_f,
                    epochs=1,
                    batch_size=len(chunk),
                    verbose=0
                )

            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        except Exception as e:
            print(f"[ERROR] Error in DQN replay: {e}")


# Optimized function to get DQN recommendation
def get_dqn_recommendation(data):
    try:
        # Fail fast if not enough data
        if len(data) < 50:
            print("[WARNING] Not enough data for DQN")
            return 0.5  # Neutral score

        # Set timeout for the entire function
        function_start_time = time.time()
        max_function_time = 60  # 60 seconds max for entire function

        # Prepare state features (last 5 days instead of 10)
        lookback = 5  # Reduced from 10
        returns = data['returns'].values[-lookback:]
        volatility = data['volatility'].values[-lookback:]

        # Check for NaN values
        if np.isnan(returns).any() or np.isnan(volatility).any():
            print("[WARNING] NaN values in state data")
            returns = np.nan_to_num(returns, nan=0.0)
            volatility = np.nan_to_num(volatility, nan=0.0)

        state = np.concatenate((returns, volatility))

        # Define action space: 0=Sell, 1=Hold, 2=Buy
        action_size = 3
        agent = DQNAgent(state_size=len(state), action_size=action_size)

        if agent.model is None:
            print("[WARNING] Failed to create DQN model")
            return 0.5  # Neutral score

        # Use a much smaller training sample - only 50 points max
        max_train_points = min(50, len(data) - (lookback + 1))

        # Use step size to skip points and reduce training time
        step_size = max(1, (len(data) - (lookback + 1)) // 50)

        # Train agent with limited historical data
        batch_counter = 0
        replay_frequency = 4  # Only replay every 4 samples

        for i in range(0, max_train_points * step_size, step_size):
            # Check timeout
            if time.time() - function_start_time > max_function_time:
                print("[WARNING] DQN function timeout reached")
                break

            # Get index with bounds checking
            idx = min(i, len(data) - (lookback + 1))

            # Extract features
            try:
                curr_returns = data['returns'].values[idx:idx + lookback]
                curr_volatility = data['volatility'].values[idx:idx + lookback]

                # Handle NaN values
                curr_returns = np.nan_to_num(curr_returns, nan=0.0)
                curr_volatility = np.nan_to_num(curr_volatility, nan=0.0)

                current_state = np.concatenate((curr_returns, curr_volatility)).reshape(1, lookback * 2)

                next_returns = data['returns'].values[idx + 1:idx + lookback + 1]
                next_volatility = data['volatility'].values[idx + 1:idx + lookback + 1]

                # Handle NaN values
                next_returns = np.nan_to_num(next_returns, nan=0.0)
                next_volatility = np.nan_to_num(next_volatility, nan=0.0)

                next_state = np.concatenate((next_returns, next_volatility)).reshape(1, lookback * 2)

                # Simplified reward function
                try:
                    reward = data['returns'].values[idx + lookback]
                    if np.isnan(reward):
                        reward = 0.0
                except:
                    reward = 0.0

                # Take action based on current state
                action = agent.act(current_state)

                # Record experience
                agent.remember(current_state, action, reward, next_state, False)

                # Train on batch less frequently to save time
                batch_counter += 1
                if batch_counter % replay_frequency == 0 and len(agent.memory) >= 8:
                    agent.replay(8)  # Use smaller batch size
            except Exception as e:
                print(f"[WARNING] Error in DQN training sample {i}: {e}")
                continue

        # Final training with all collected data if time permits
        if time.time() - function_start_time <= max_function_time and len(agent.memory) >= 16:
            try:
                agent.replay(16)
            except Exception as e:
                print(f"[WARNING] Error in final DQN training: {e}")

        # Get recommendation
        state = state.reshape(1, len(state))
        try:
            action = agent.act(state)
            # Convert action to score: 0=0.0, 1=0.5, 2=1.0
            dqn_score = action / 2
            return dqn_score
        except:
            print("[WARNING] Error getting DQN action, returning neutral score")
            return 0.5

    except Exception as e:
        print(f"[ERROR] Error in DQN recommendation: {e}")
        traceback.print_exc()
        return 0.5  # Neutral score


# Calculate Sigma metric combining all analysis tools with timeout
def calculate_sigma(data):
    try:
        # Set a maximum execution time for the entire function
        max_execution_time = 180  # 3 minutes max
        start_time = time.time()

        # 1. Calculate technical indicators
        indicators_df = calculate_technical_indicators(data)
        if indicators_df is None or len(indicators_df) < 30:
            print("[WARNING] Technical indicators calculation failed or insufficient data")
            return None

        # Check time after expensive calculation
        if time.time() - start_time > max_execution_time * 0.3:  # If 30% of time already used
            print("[WARNING] Technical indicators calculation took too long")
            # Skip PCA and other complex calculations

            # Get more technical signals from the latest data point
            latest = indicators_df.iloc[-1]

            # Calculate a simplified Sigma based only on technical indicators
            traditional_volatility = indicators_df['volatility'].iloc[-1] if not np.isnan(
                indicators_df['volatility'].iloc[-1]) else 0
            rsi = latest['RSI'] if not np.isnan(latest['RSI']) else 50
            rsi_signal = (max(0, min(100, rsi)) - 30) / 70
            rsi_signal = max(0, min(1, rsi_signal))

            # Simple MACD signal
            macd = latest['MACD'] if not np.isnan(latest['MACD']) else 0
            macd_signal = np.tanh(macd * 10)

            # Simple SMA trend
            sma20 = latest['SMA20'] if not np.isnan(latest['SMA20']) else 1
            sma50 = latest['SMA50'] if not np.isnan(latest['SMA50']) else 1
            if abs(sma50) < 1e-6:
                sma_trend = 0
            else:
                sma_trend = (sma20 / sma50 - 1)
            sma_signal = np.tanh(sma_trend * 10)

            # Calculate simplified Sigma
            simplified_sigma = (
                    0.3 * min(1, traditional_volatility * 25) +
                    0.3 * rsi_signal +
                    0.2 * (macd_signal + 1) / 2 +
                    0.2 * (sma_signal + 1) / 2
            )

            simplified_sigma = max(0, min(1, simplified_sigma))

            # Return simplified analysis
            analysis_details = {
                "sigma": simplified_sigma,
                "traditional_volatility": traditional_volatility,
                "lstm_prediction": 0,
                "dqn_recommendation": 0.5,
                "rsi_signal": rsi,
                "macd": macd,
                "sma_trend": sma_trend,
                "bb_position": 0.5,
                "volume_change": 0,
                "pca_variance": [],
                "last_price": latest['4. close'] if not np.isnan(latest['4. close']) else 0,
                "simplified": True
            }

            return analysis_details

        # 2. Apply PCA to reduce feature dimensionality - make it optional
        pca_results = None
        pca_variance = []

        # Only do PCA if we have enough time
        if time.time() - start_time < max_execution_time * 0.4:  # If less than 40% of time used
            try:
                # Use last 30 days or all available data, whichever is smaller
                lookback_period = min(30, len(indicators_df))
                pca_results, pca_variance = apply_pca(indicators_df.iloc[-lookback_period:])
            except:
                print("[WARNING] PCA calculation failed, continuing without it")
                pca_variance = []
        else:
            print("[WARNING] Skipping PCA calculation due to time constraints")

        # 3. Get LSTM volatility prediction with a timeout
        lstm_prediction = 0
        if time.time() - start_time < max_execution_time * 0.6:  # If less than 60% of time used
            lstm_prediction = predict_with_lstm(data)
            print(f"[DEBUG] LSTM prediction: {lstm_prediction}")
        else:
            print("[WARNING] Skipping LSTM prediction due to time constraints")

        # 4. Get DQN recommendation with a timeout
        dqn_recommendation = 0.5  # Default neutral
        if time.time() - start_time < max_execution_time * 0.8:  # If less than 80% of time used
            dqn_recommendation = get_dqn_recommendation(indicators_df)
            print(f"[DEBUG] DQN recommendation: {dqn_recommendation}")
        else:
            print("[WARNING] Skipping DQN recommendation due to time constraints")

        # 5. Calculate traditional volatility (basic and reliable)
        traditional_volatility = indicators_df['volatility'].iloc[-1]
        if np.isnan(traditional_volatility):
            traditional_volatility = 0

        # Get more technical signals from the latest data point
        latest = indicators_df.iloc[-1]

        # RSI signal (0-1 scale, higher means oversold/bullish)
        rsi = latest['RSI'] if not np.isnan(latest['RSI']) else 50
        rsi_signal = (max(0, min(100, rsi)) - 30) / 70
        rsi_signal = max(0, min(1, rsi_signal))  # Ensure between 0 and 1

        # MACD signal (-1 to 1 scale)
        macd = latest['MACD'] if not np.isnan(latest['MACD']) else 0
        macd_signal = np.tanh(macd * 10)

        # SMA trend signal (-1 to 1 scale)
        sma20 = latest['SMA20'] if not np.isnan(latest['SMA20']) else 1
        sma50 = latest['SMA50'] if not np.isnan(latest['SMA50']) else 1
        if abs(sma50) < 1e-6:  # Avoid division by zero
            sma_trend = 0
        else:
            sma_trend = (sma20 / sma50 - 1)
        sma_signal = np.tanh(sma_trend * 10)

        # Bollinger Band position (0-1 scale)
        bb_upper = latest['BB_upper'] if not np.isnan(latest['BB_upper']) else 0
        bb_lower = latest['BB_lower'] if not np.isnan(latest['BB_lower']) else 0
        close_price = latest['4. close'] if not np.isnan(latest['4. close']) else 0

        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            bb_position = (close_price - bb_lower) / bb_range
        else:
            bb_position = 0.5
        bb_position = max(0, min(1, bb_position))  # Ensure between 0 and 1

        # Volume signal (0-1 scale)
        volume_change = latest['volume_change'] if not np.isnan(latest['volume_change']) else 0
        volume_signal = min(1, max(0, (volume_change + 0.1) / 0.2))

        # Calculate final Sigma score (0-1 scale)
        # Combine all signals with weights, but adapt based on what was calculated
        if lstm_prediction > 0 and dqn_recommendation != 0.5:
            # Full calculation
            sigma = (
                    0.20 * min(1, traditional_volatility * 25) +
                    0.15 * lstm_prediction +
                    0.15 * dqn_recommendation +
                    0.15 * rsi_signal +
                    0.15 * (macd_signal + 1) / 2 +
                    0.10 * (sma_signal + 1) / 2 +
                    0.05 * bb_position +
                    0.05 * volume_signal
            )
        else:
            # Simplified calculation with reweighted components
            sigma = (
                    0.25 * min(1, traditional_volatility * 25) +
                    0.25 * rsi_signal +
                    0.20 * (macd_signal + 1) / 2 +
                    0.15 * (sma_signal + 1) / 2 +
                    0.10 * bb_position +
                    0.05 * volume_signal
            )

        # Ensure sigma is between 0 and 1
        sigma = max(0, min(1, sigma))

        # For detailed analysis, return a dictionary
        analysis_details = {
            "sigma": sigma,
            "traditional_volatility": traditional_volatility,
            "lstm_prediction": lstm_prediction,
            "dqn_recommendation": dqn_recommendation,
            "rsi_signal": rsi,
            "macd": macd,
            "sma_trend": sma_trend,
            "bb_position": bb_position,
            "volume_change": volume_change,
            "pca_variance": pca_variance if pca_variance is not None else [],
            "last_price": latest['4. close'] if not np.isnan(latest['4. close']) else 0,
            "simplified": lstm_prediction == 0 or dqn_recommendation == 0.5
        }

        return analysis_details
    except Exception as e:
        print(f"[ERROR] Error calculating Sigma: {e}")
        traceback.print_exc()
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


# Create or initialize the output file
def initialize_output_file():
    try:
        with open(OUTPUT_FILE, "w") as file:
            file.write("===== XTB STOCK ANALYSIS DATABASE =====\n")
            file.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write("FORMAT: TICKER | PRICE | SIGMA | RECOMMENDATION\n")
            file.write("----------------------------------------\n")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize output file: {e}")
        return False


# Append stock analysis result to the output file
def append_stock_result(symbol, price, sigma, recommendation):
    try:
        with open(OUTPUT_FILE, "a") as file:
            # Format: TICKER | PRICE | SIGMA | RECOMMENDATION
            file.write(f"{symbol} | ${price:.2f} | {sigma:.5f} | {recommendation}\n")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to append result for {symbol}: {e}")
        return False


# Function to analyze a single stock with timeout
def analyze_stock(client, symbol, max_time=MAX_EXECUTION_TIME_PER_STOCK):
    print(f"[INFO] Analyzing stock: {symbol}")
    start_time = time.time()

    try:
        # Get historical data
        data = get_stock_data(client, symbol)

        # Check if we got valid data
        if data is None or len(data) < 60:
            print(f"[WARNING] Insufficient data for {symbol}")
            return None

        # Check if we're still within time limit
        if time.time() - start_time > max_time * 0.5:  # If 50% of time already used
            print(f"[WARNING] Data retrieval for {symbol} took too long")
            return None

        # Calculate Sigma
        analysis = calculate_sigma(data)

        if analysis is None:
            print(f"[WARNING] Failed to calculate Sigma for {symbol}")
            return None

        # Get recommendation
        sigma = analysis["sigma"]
        recommendation = get_sigma_recommendation(sigma)
        price = analysis["last_price"]

        print(f"[INFO] Analysis complete for {symbol}: Sigma={sigma:.5f}, Recommendation={recommendation}")

        # Return the result
        return {
            "symbol": symbol,
            "price": price,
            "sigma": sigma,
            "recommendation": recommendation
        }

    except Exception as e:
        print(f"[ERROR] Error analyzing {symbol}: {e}")
        traceback.print_exc()
        return None
    finally:
        elapsed_time = time.time() - start_time
        print(f"[INFO] Analysis of {symbol} took {elapsed_time:.1f} seconds")


# Process stocks in batches
def process_stocks_in_batches(stocks, batch_size=MAX_STOCKS_PER_BATCH):
    print(f"[INFO] Starting batch processing of {len(stocks)} stocks")

    # Initialize output file
    if not initialize_output_file():
        print("[ERROR] Failed to initialize output file. Aborting.")
        return False

    # Track overall progress
    total_analyzed = 0
    total_successful = 0

    # Set overall timeout
    overall_start_time = time.time()

    # Process in batches
    for i in range(0, len(stocks), batch_size):
        batch = stocks[i:i + batch_size]
        print(f"[INFO] Processing batch {i // batch_size + 1}/{(len(stocks) + batch_size - 1) // batch_size}")

        # Check overall timeout
        if time.time() - overall_start_time > MAX_TOTAL_RUNTIME:
            print(f"[WARNING] Maximum total runtime ({MAX_TOTAL_RUNTIME / 3600:.1f} hours) reached. Stopping.")
            break

        # Connect to XTB for this batch
        client = XTBClient()
        connection_success = client.connect()

        if not connection_success:
            print("[ERROR] Failed to connect to XTB for this batch. Trying again after delay.")
            time.sleep(BATCH_DELAY * 2)  # Longer delay after connection failure
            continue

        # Process each stock in the batch
        for stock in batch:
            symbol = stock["symbol"]

            # Check overall timeout
            if time.time() - overall_start_time > MAX_TOTAL_RUNTIME:
                print(f"[WARNING] Maximum total runtime reached during batch. Stopping.")
                break

            # Skip if we're not logged in
            if not client.logged_in:
                print(f"[WARNING] Not logged in. Reconnecting...")
                client.disconnect()
                time.sleep(5)
                connection_success = client.connect()
                if not connection_success:
                    print("[ERROR] Reconnection failed. Skipping rest of batch.")
                    break

            # Analyze the stock
            result = analyze_stock(client, symbol)
            total_analyzed += 1

            # If analysis successful, save the result
            if result:
                append_stock_result(
                    result["symbol"],
                    result["price"],
                    result["sigma"],
                    result["recommendation"]
                )
                total_successful += 1

            # Print progress
            progress = (total_analyzed / len(stocks)) * 100
            success_rate = (total_successful / total_analyzed) * 100 if total_analyzed > 0 else 0
            print(
                f"[INFO] Progress: {progress:.1f}% ({total_analyzed}/{len(stocks)}), Success rate: {success_rate:.1f}%")

        # Disconnect after batch is complete
        client.disconnect()

        # Wait between batches to avoid rate limits
        if i + batch_size < len(stocks):  # If not the last batch
            print(f"[INFO] Batch complete. Waiting {BATCH_DELAY} seconds before next batch...")
            time.sleep(BATCH_DELAY)

    # Final report
    print(
        f"[INFO] Analysis complete! Analyzed {total_analyzed}/{len(stocks)} stocks with {total_successful} successful analyses.")
    print(f"[INFO] Results saved to {OUTPUT_FILE}")

    return True


# Main function to run the entire database analysis
def analyze_entire_database():
    print("[INFO] Starting analysis of entire XTB stock database")

    # Connect to XTB
    client = XTBClient()

    if not client.connect():
        print("[ERROR] Failed to connect to XTB API. Exiting.")
        return False

    try:
        # Get all stock symbols
        all_stocks = get_all_stock_symbols(client)

        if not all_stocks or len(all_stocks) == 0:
            print("[ERROR] Failed to retrieve stock symbols or no stocks found.")
            client.disconnect()
            return False

        print(f"[INFO] Retrieved {len(all_stocks)} stock symbols for analysis")

        # Disconnect since we'll reconnect in batches
        client.disconnect()

        # Filter stocks if needed
        # You can add filters here to focus on specific markets, categories, etc.
        filtered_stocks = all_stocks
        print(f"[INFO] Filtered to {len(filtered_stocks)} stocks for analysis")

        # Process all stocks in batches
        success = process_stocks_in_batches(filtered_stocks)

        return success

    except Exception as e:
        print(f"[ERROR] Error in database analysis: {e}")
        traceback.print_exc()
        return False
    finally:
        # Ensure we disconnect
        try:
            if client and hasattr(client, 'disconnect'):
                client.disconnect()
        except:
            pass


# Run the analysis if this script is executed directly
if __name__ == "__main__":
    try:
        print("===== XTB STOCK DATABASE ANALYZER =====")
        analyze_entire_database()
    except KeyboardInterrupt:
        print("\n[INFO] Analysis stopped by user")
    except Exception as e:
        print(f"[ERROR] Unhandled exception: {e}")
        traceback.print_exc()