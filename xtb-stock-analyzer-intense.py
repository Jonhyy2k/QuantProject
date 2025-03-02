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

# Global settings for batch processing - optimized for M1 iMac
MAX_STOCKS_PER_BATCH = 250  # Increased batch size for more powerful CPU
BATCH_DELAY = 15  # Reduced delay between batches since processing is faster
MAX_EXECUTION_TIME_PER_STOCK = 300  # 5 minutes max per stock for more thorough analysis
MAX_TOTAL_RUNTIME = 240 * 3600  # 240 hours (10 days) maximum total runtime


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


# Enhanced data preparation for LSTM prediction with additional features
def prepare_lstm_data(data, time_steps=50):  # Increased from 30
    try:
        # Check if we have enough data
        if len(data) < time_steps + 10:
            print(f"[WARNING] Not enough data for LSTM: {len(data)} < {time_steps + 10}")
            return None, None, None

        # Use multiple features instead of just closing prices
        # Include price, volume, and technical indicators if available
        features = []
        
        # Always include closing price
        features.append(data['4. close'].values)
        
        # Include volume if available with appropriate scaling
        if 'volume' in data.columns:
            # Log transform volume to reduce scale differences
            log_volume = np.log1p(data['volume'].values)
            features.append(log_volume)
        
        # Include volatility if available
        if 'volatility' in data.columns:
            features.append(data['volatility'].values)
            
        # Include RSI if available
        if 'RSI' in data.columns:
            # Normalize RSI to 0-1 scale
            normalized_rsi = data['RSI'].values / 100
            features.append(normalized_rsi)
            
        # Include MACD if available
        if 'MACD' in data.columns:
            # Normalize MACD using tanh for -1 to 1 range
            normalized_macd = np.tanh(data['MACD'].values / 5)
            features.append(normalized_macd)
            
        # Stack features
        feature_array = np.column_stack(features)
        
        # Check for NaN values across all features
        if np.isnan(feature_array).any():
            print(f"[WARNING] NaN values in features, filling with forward fill")
            # Convert to DataFrame for easier handling of NaNs
            temp_df = pd.DataFrame(feature_array)
            # Fill NaN values
            temp_df = temp_df.fillna(method='ffill').fillna(method='bfill')
            feature_array = temp_df.values

        # Normalize the data
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_array)

        # Create sequences with all features
        X, y = [], []
        # Target is still the closing price (first feature)
        for i in range(len(scaled_features) - time_steps):
            X.append(scaled_features[i:i + time_steps])
            # For prediction target, use only the closing price column (index 0)
            y.append(scaled_features[i + time_steps, 0:1])

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Check shapes
        print(f"[DEBUG] Enhanced LSTM data shapes: X={X.shape}, y={y.shape}")

        return X, y, scaler
    except Exception as e:
        print(f"[ERROR] Error preparing enhanced LSTM data: {e}")
        traceback.print_exc()
        # Fallback to simpler preparation if enhanced fails
        try:
            print(f"[WARNING] Falling back to simple price-only LSTM preparation")
            # Get closing prices only
            prices = data['4. close'].values
            
            # Handle NaN values
            if np.isnan(prices).any():
                prices = pd.Series(prices).fillna(method='ffill').fillna(method='bfill').values
                
            # Reshape and scale
            prices_2d = prices.reshape(-1, 1)
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
            
            print(f"[DEBUG] Fallback LSTM data shapes: X={X.shape}, y={y.shape}")
            return X, y, scaler
            
        except Exception as e2:
            print(f"[ERROR] Fallback LSTM data preparation also failed: {e2}")
            return None, None, None


# Enhanced LSTM model for volatility prediction - optimized for M1
def build_lstm_model(input_shape):
    try:
        # More sophisticated architecture for better prediction accuracy
        inputs = Input(shape=input_shape)
        x = LSTM(64, return_sequences=True)(inputs)  # More units in first layer
        x = Dropout(0.2)(x)  # Add dropout for regularization
        x = LSTM(64, return_sequences=False)(x)  # Second LSTM layer
        x = Dense(32, activation='relu')(x)  # Additional dense layer
        x = Dropout(0.1)(x)  # More regularization
        outputs = Dense(1)(x)  # Output layer
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        return model
    except Exception as e:
        print(f"[ERROR] Error building LSTM model: {e}")
        traceback.print_exc()
        
        # Fallback to simpler model if complex one fails
        try:
            inputs = Input(shape=input_shape)
            x = LSTM(48, return_sequences=False)(inputs)
            outputs = Dense(1)(x)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer="adam", loss="mse")
            return model
        except Exception as e2:
            print(f"[ERROR] Fallback LSTM model also failed: {e2}")
            return None


# Enhanced LSTM model training and prediction with extended processing time
def predict_with_lstm(data):
    try:
        # Set a maximum execution time - increased for more thorough training
        max_execution_time = 120  # 120 seconds max (2 minutes) 
        start_time = time.time()

        # Require less data to attempt prediction
        if len(data) < 60:
            print("[WARNING] Not enough data for LSTM model")
            return 0

        # Use a larger window for more context
        time_steps = 50  # Increased from 30 for better prediction accuracy
        
        # Prepare data
        X, y, scaler = prepare_lstm_data(data, time_steps=time_steps)
        if X is None or y is None or scaler is None:
            print("[WARNING] Failed to prepare LSTM data")
            return 0

        # More lenient on required data size
        if len(X) < 8:
            print(f"[WARNING] Not enough data after preparation: {len(X)}")
            return 0

        # Build enhanced model
        model = build_lstm_model((X.shape[1], 1))
        if model is None:
            print("[WARNING] Failed to build LSTM model")
            return 0

        # Use more training data
        max_samples = 500  # Increased from 200
        if len(X) > max_samples:
            # Use evenly spaced samples to get good representation
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
                # Use a better fallback prediction based on recent volatility
                return data['volatility'].iloc[-10:].mean() / data['volatility'].iloc[-30:].mean()

            # Train model with more epochs and patience
            early_stop = EarlyStopping(monitor='loss', patience=3, verbose=0)  # Increased patience

            # Set parameters for better training
            model.fit(
                X_train, y_train,
                epochs=15,  # Increased from 5
                batch_size=32,
                callbacks=[early_stop],
                verbose=0,
                shuffle=True
            )
            
            # Extra training round with lower learning rate for fine-tuning
            if time.time() - start_time < max_execution_time * 0.7:
                model.optimizer.lr = model.optimizer.lr * 0.5  # Reduce learning rate
                model.fit(
                    X_train, y_train, 
                    epochs=10,
                    batch_size=32,
                    verbose=0,
                    shuffle=True
                )
                
        except Exception as e:
            print(f"[ERROR] LSTM model training failed: {e}")
            return 0

        # Make prediction for future volatility
        try:
            # Check time again
            if time.time() - start_time > max_execution_time:
                print("[WARNING] LSTM execution time limit reached before prediction")
                return 0.5  # Return a neutral value

            # Use the last few sequences for better prediction stability
            num_pred_samples = min(5, len(X))
            predictions = []
            
            for i in range(num_pred_samples):
                seq_idx = len(X) - i - 1
                sequence = X[seq_idx].reshape(1, X.shape[1], 1)
                pred = model.predict(sequence, verbose=0)[0][0]
                predictions.append(pred)
            
            avg_prediction = np.mean(predictions)
            last_actual = np.mean(y[-num_pred_samples:])

            # Avoid division by zero
            if abs(last_actual) < 1e-6:
                predicted_volatility_change = abs(avg_prediction)
            else:
                predicted_volatility_change = abs((avg_prediction - last_actual) / last_actual)

            print(f"[DEBUG] LSTM prediction: {predicted_volatility_change}")
            return min(1.0, predicted_volatility_change)  # Cap at 1.0
            
        except Exception as e:
            print(f"[ERROR] LSTM prediction failed: {e}")
            return 0
    except Exception as e:
        print(f"[ERROR] Error in LSTM prediction: {e}")
        traceback.print_exc()
        return 0


# Enhanced DQN Agent implementation for more accurate predictions
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)  # Increased memory size for better learning
        self.gamma = 0.97  # Increased discount factor for more future-focused decisions
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Lower minimum for better exploitation
        self.epsilon_decay = 0.95  # Slower decay for better exploration
        self.model = self._build_model()
        self.target_model = self._build_model()  # Separate target network for more stable learning
        self.target_update_counter = 0
        self.target_update_freq = 10  # Update target network every 10 training sessions
        self.max_training_time = 60  # Increased maximum training time (1 minute)

    def _build_model(self):
        try:
            # More complex model architecture for better learning
            model = Sequential([
                Dense(128, activation="relu", input_shape=(self.state_size,)),
                Dropout(0.2),  # Add dropout for regularization
                Dense(128, activation="relu"),
                Dropout(0.2),  # More dropout
                Dense(64, activation="relu"),
                Dense(self.action_size, activation="linear")
            ])
            model.compile(optimizer="adam", loss="mse")
            return model
        except Exception as e:
            print(f"[ERROR] Error building enhanced DQN model: {e}")
            # Fallback to simpler model if enhanced fails
            try:
                model = Sequential([
                    Dense(64, activation="relu", input_shape=(self.state_size,)),
                    Dense(64, activation="relu"),
                    Dense(self.action_size, activation="linear")
                ])
                model.compile(optimizer="adam", loss="mse")
                return model
            except Exception as e2:
                print(f"[ERROR] Error building intermediate DQN model: {e2}")
                # Even simpler fallback model
                try:
                    model = Sequential([
                        Dense(32, activation="relu", input_shape=(self.state_size,)),
                        Dense(self.action_size, activation="linear")
                    ])
                    model.compile(optimizer="adam", loss="mse")
                    return model
                except Exception as e3:
                    print(f"[ERROR] Error building simplest DQN model: {e3}")
                    return None
    
    # Update target model (for more stable learning)
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        print("[DEBUG] DQN target model updated")

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
            # Use larger batch sizes for more stable learning
            actual_batch_size = min(batch_size, len(self.memory))
            minibatch = random.sample(self.memory, actual_batch_size)

            # Process in reasonable chunks for better performance
            chunk_size = 32  # Increased from 8 for better batch learning
            
            for i in range(0, len(minibatch), chunk_size):
                chunk = minibatch[i:i + chunk_size]

                # Check timeout
                if time.time() - start_time > self.max_training_time:
                    print("[WARNING] DQN training timeout reached")
                    break

                # Process chunk
                states = np.vstack([x[0] for x in chunk])
                
                # Use the target network for more stable learning
                next_states = np.vstack([x[3] for x in chunk])
                actions = np.array([x[1] for x in chunk])
                rewards = np.array([x[2] for x in chunk])
                dones = np.array([x[4] for x in chunk])
                
                # Current Q values
                targets = self.model.predict(states, verbose=0)
                
                # Get next Q values from target model
                next_q_values = self.target_model.predict(next_states, verbose=0)
                
                # Update Q values - more efficient vectorized approach
                for j in range(len(chunk)):
                    if dones[j]:
                        targets[j, actions[j]] = rewards[j]
                    else:
                        targets[j, actions[j]] = rewards[j] + self.gamma * np.max(next_q_values[j])

                # Fit with more epochs for better learning
                self.model.fit(
                    states,
                    targets,
                    epochs=3,  # Increased from 1
                    batch_size=len(chunk),
                    verbose=0
                )

            # Update epsilon with a more gradual decay
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            # Update target network periodically
            self.target_update_counter += 1
            if self.target_update_counter >= self.target_update_freq:
                self.update_target_model()
                self.target_update_counter = 0

        except Exception as e:
            print(f"[ERROR] Error in DQN replay: {e}")
            traceback.print_exc()


# Enhanced DQN recommendation with more features and training time
def get_dqn_recommendation(data):
    try:
        # Slightly more lenient on required data
        if len(data) < 40:
            print("[WARNING] Not enough data for DQN")
            return 0.5  # Neutral score

        # Set timeout for the entire function - increased for more thorough training
        function_start_time = time.time()
        max_function_time = 120  # 120 seconds (2 minutes) max for entire function

        # Prepare state features with more historical context
        lookback = 10  # Increased from 5 for better historical context
        
        # Extract more features for a richer state representation
        features = []
        features.append(data['returns'].values[-lookback:])  # Returns
        features.append(data['volatility'].values[-lookback:])  # Volatility
        
        # Add RSI if available (normalized)
        if 'RSI' in data.columns:
            rsi = data['RSI'].values[-lookback:] / 100  # Normalize to 0-1
            features.append(rsi)
            
        # Add MACD if available
        if 'MACD' in data.columns:
            # Normalize MACD with tanh to -1 to 1 range
            macd = np.tanh(data['MACD'].values[-lookback:] / 5)
            features.append(macd)
            
        # Add SMA trend if available
        if 'SMA20' in data.columns and 'SMA50' in data.columns:
            sma20 = data['SMA20'].values[-lookback:]
            sma50 = data['SMA50'].values[-lookback:]
            # Calculate relative position of SMA
            with np.errstate(divide='ignore', invalid='ignore'):
                sma_ratio = np.where(sma50 != 0, sma20 / sma50, 1.0)
            sma_ratio = np.nan_to_num(sma_ratio, nan=1.0)
            # Normalize around 1.0 (above 1 = bullish, below 1 = bearish)
            sma_trend = np.tanh((sma_ratio - 1.0) * 5)
            features.append(sma_trend)
        
        # Stack all features into the state
        features = [np.nan_to_num(f, nan=0.0) for f in features]  # Handle NaNs
        state = np.concatenate(features)

        # Define action space: 0=Sell, 1=Hold, 2=Buy
        action_size = 3
        agent = DQNAgent(state_size=len(state), action_size=action_size)

        if agent.model is None:
            print("[WARNING] Failed to create DQN model")
            return 0.5  # Neutral score

        # Use more training data for better learning
        max_train_points = min(200, len(data) - (lookback + 1))  # Increased from 50

        # Use appropriate step size to get good coverage of data
        step_size = max(1, (len(data) - (lookback + 1)) // 200)  # Adjusted for more points

        # Train agent with historical data
        batch_counter = 0
        replay_frequency = 2  # More frequent replay (was 4)

        # First pass: collect experiences without training to populate memory
        print("[DEBUG] DQN collecting initial experiences...")
        for i in range(0, max_train_points * step_size, step_size):
            # Check timeout
            if time.time() - function_start_time > max_function_time * 0.3:  # Use 30% of time for collection
                print("[WARNING] DQN experience collection timeout reached")
                break

            # Get index with bounds checking
            idx = min(i, len(data) - (lookback + 1))

            # Extract features for current state
            try:
                # Create feature list for this time point
                curr_features = []
                curr_features.append(data['returns'].values[idx:idx + lookback])
                curr_features.append(data['volatility'].values[idx:idx + lookback])
                
                if 'RSI' in data.columns:
                    curr_rsi = data['RSI'].values[idx:idx + lookback] / 100
                    curr_features.append(curr_rsi)
                    
                if 'MACD' in data.columns:
                    curr_macd = np.tanh(data['MACD'].values[idx:idx + lookback] / 5)
                    curr_features.append(curr_macd)
                    
                if 'SMA20' in data.columns and 'SMA50' in data.columns:
                    curr_sma20 = data['SMA20'].values[idx:idx + lookback]
                    curr_sma50 = data['SMA50'].values[idx:idx + lookback]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        curr_sma_ratio = np.where(curr_sma50 != 0, curr_sma20 / curr_sma50, 1.0)
                    curr_sma_ratio = np.nan_to_num(curr_sma_ratio, nan=1.0)
                    curr_sma_trend = np.tanh((curr_sma_ratio - 1.0) * 5)
                    curr_features.append(curr_sma_trend)

                # Handle NaN values and create current state
                curr_features = [np.nan_to_num(f, nan=0.0) for f in curr_features]
                current_state = np.concatenate(curr_features).reshape(1, len(state))

                # Get next state features (one step ahead)
                next_idx = idx + 1
                if next_idx + lookback <= len(data):
                    next_features = []
                    next_features.append(data['returns'].values[next_idx:next_idx + lookback])
                    next_features.append(data['volatility'].values[next_idx:next_idx + lookback])
                    
                    if 'RSI' in data.columns:
                        next_rsi = data['RSI'].values[next_idx:next_idx + lookback] / 100
                        next_features.append(next_rsi)
                        
                    if 'MACD' in data.columns:
                        next_macd = np.tanh(data['MACD'].values[next_idx:next_idx + lookback] / 5)
                        next_features.append(next_macd)
                        
                    if 'SMA20' in data.columns and 'SMA50' in data.columns:
                        next_sma20 = data['SMA20'].values[next_idx:next_idx + lookback]
                        next_sma50 = data['SMA50'].values[next_idx:next_idx + lookback]
                        with np.errstate(divide='ignore', invalid='ignore'):
                            next_sma_ratio = np.where(next_sma50 != 0, next_sma20 / next_sma50, 1.0)
                        next_sma_ratio = np.nan_to_num(next_sma_ratio, nan=1.0)
                        next_sma_trend = np.tanh((next_sma_ratio - 1.0) * 5)
                        next_features.append(next_sma_trend)
                    
                    # Handle NaN values and create next state
                    next_features = [np.nan_to_num(f, nan=0.0) for f in next_features]
                    next_state = np.concatenate(next_features).reshape(1, len(state))
                    
                    # Improved reward function - based on return and also trend direction
                    try:
                        # Base reward on return
                        price_return = data['returns'].values[next_idx + lookback - 1]
                        
                        # Add trend component to reward
                        trend_component = 0
                        if 'SMA20' in data.columns and 'SMA50' in data.columns:
                            trend = next_sma_trend[-1]  # Use last value of SMA trend
                            trend_component = trend * 0.01  # Small adjustment based on trend
                            
                        # Combine components
                        reward = price_return + trend_component
                        
                        if np.isnan(reward):
                            reward = 0.0
                    except:
                        reward = 0.0

                    # Take action based on current state
                    action = agent.act(current_state)

                    # Record experience
                    is_terminal = (next_idx + lookback >= len(data) - 1)
                    agent.remember(current_state, action, reward, next_state, is_terminal)
                
            except Exception as e:
                print(f"[WARNING] Error in DQN experience collection sample {i}: {e}")
                continue

        # Second pass: train on collected experiences
        print(f"[DEBUG] DQN training on {len(agent.memory)} experiences...")
        
        # Perform training in several batches with increasing batch size
        if len(agent.memory) >= 32:
            # Multiple training rounds with different batch sizes
            batch_sizes = [32, 64, 128, 256]
            for batch_size in batch_sizes:
                if batch_size <= len(agent.memory):
                    # Check if we still have time
                    if time.time() - function_start_time > max_function_time * 0.7:
                        print("[WARNING] DQN training timeout reached during main training")
                        break
                        
                    print(f"[DEBUG] DQN training with batch size {batch_size}")
                    agent.replay(batch_size)
                    
            # Final training with full memory if time permits
            if time.time() - function_start_time <= max_function_time * 0.9 and len(agent.memory) >= 512:
                print("[DEBUG] DQN final training pass")
                agent.replay(min(512, len(agent.memory)))

        # Get recommendation with averaging for more stability
        try:
            # Make multiple predictions with slight perturbations for robustness
            num_predictions = 5
            actions = []
            
            for _ in range(num_predictions):
                # Apply small random noise to state
                perturbed_state = state.copy()
                perturbed_state += np.random.normal(0, 0.02, size=state.shape)  # 2% noise
                perturbed_state = perturbed_state.reshape(1, len(state))
                
                # Get action
                action = agent.act(perturbed_state)
                actions.append(action)
                
            # Use most common action (majority vote)
            most_common_action = max(set(actions), key=actions.count)
            
            # Convert action to score: 0=0.0, 1=0.5, 2=1.0
            dqn_score = most_common_action / 2
            return dqn_score
            
        except Exception as e:
            print(f"[WARNING] Error getting DQN action: {e}")
            return 0.5  # Neutral score

    except Exception as e:
        print(f"[ERROR] Error in DQN recommendation: {e}")
        traceback.print_exc()
        return 0.5  # Neutral score


# Enhanced Sigma metric calculation with more sophisticated analysis models
def calculate_sigma(data):
    try:
        # Set a maximum execution time for the entire function - increased for more thorough analysis
        max_execution_time = 300  # 5 minutes max
        start_time = time.time()

        # 1. Calculate technical indicators
        indicators_df = calculate_technical_indicators(data)
        if indicators_df is None or len(indicators_df) < 30:
            print("[WARNING] Technical indicators calculation failed or insufficient data")
            return None

        # Intermediate check time after technical indicators calculation
        # More lenient threshold to allow continuing to advanced analysis
        if time.time() - start_time > max_execution_time * 0.2:  # If 20% of time already used
            print("[WARNING] Technical indicators calculation took longer than expected")
            print("[INFO] Continuing with all analysis methods, but will monitor time closely")

        # 2. Apply PCA to reduce feature dimensionality - more likely to include
        pca_results = None
        pca_variance = []
        pca_components = None

        # Only skip PCA if we're very constrained on time
        if time.time() - start_time < max_execution_time * 0.5:  # More generous time allocation
            try:
                # Use more historical data for PCA
                lookback_period = min(60, len(indicators_df))  # Increased from 30
                pca_results, pca_variance = apply_pca(indicators_df.iloc[-lookback_period:])
                
                if pca_results is not None:
                    # Store pca components for possible use in final sigma calculation
                    pca_components = pca_results.iloc[-1].values
                    print(f"[DEBUG] PCA components for latest datapoint: {pca_components}")
            except Exception as e:
                print(f"[WARNING] PCA calculation failed: {e}, continuing without it")
                pca_variance = []
        else:
            print("[WARNING] Skipping PCA calculation due to significant time constraints")

        # 3. Get LSTM volatility prediction
        lstm_prediction = 0
        if time.time() - start_time < max_execution_time * 0.7:  # More generous time allocation
            lstm_prediction = predict_with_lstm(data)
            print(f"[DEBUG] LSTM prediction: {lstm_prediction}")
        else:
            print("[WARNING] Skipping LSTM prediction due to time constraints")

        # 4. Get DQN recommendation
        dqn_recommendation = 0.5  # Default neutral
        if time.time() - start_time < max_execution_time * 0.8:  # More generous time allocation
            dqn_recommendation = get_dqn_recommendation(indicators_df)
            print(f"[DEBUG] DQN recommendation: {dqn_recommendation}")
        else:
            print("[WARNING] Skipping DQN recommendation due to time constraints")

        # Get latest technical indicator values
        latest = indicators_df.iloc[-1]

        # 5. Calculate traditional volatility (basic and reliable)
        traditional_volatility = indicators_df['volatility'].iloc[-1]
        if np.isnan(traditional_volatility):
            traditional_volatility = 0

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
        
        # 6. Enhanced momentum indicators
        # Calculate short-term momentum (last 10 days vs previous 10 days)
        try:
            recent_returns = indicators_df['returns'].iloc[-10:].mean()
            previous_returns = indicators_df['returns'].iloc[-20:-10].mean()
            momentum_signal = np.tanh((recent_returns - previous_returns) * 20)  # Scale to approx -1 to 1
            momentum_signal = (momentum_signal + 1) / 2  # Convert to 0-1 scale
        except:
            momentum_signal = 0.5  # Neutral
            
        # 7. Calculate volatility trend (rising or falling volatility)
        try:
            recent_vol = indicators_df['volatility'].iloc[-5:].mean()
            previous_vol = indicators_df['volatility'].iloc[-15:-5].mean()
            vol_trend = (recent_vol / previous_vol) if previous_vol > 0 else 1.0
            vol_trend_signal = min(1, max(0, 1.5 - vol_trend))  # Lower is better (less volatility increase)
        except:
            vol_trend_signal = 0.5  # Neutral

        # Calculate final Sigma score (0-1 scale) with all available signals
        # Adjust weighting based on what was calculated and significance
        
        # Define base components dictionary for readability and debugging
        components = {
            "traditional_volatility": min(1, traditional_volatility * 25),
            "rsi": rsi_signal,
            "macd": (macd_signal + 1) / 2,  # Convert from -1:1 to 0:1
            "sma_trend": (sma_signal + 1) / 2,  # Convert from -1:1 to 0:1
            "bb_position": bb_position,
            "volume": volume_signal,
            "momentum": momentum_signal,
            "vol_trend": vol_trend_signal,
            "lstm": lstm_prediction,
            "dqn": dqn_recommendation
        }
        
        print(f"[DEBUG] Sigma components: {components}")
        
        # Adaptive weighting based on what's available
        if lstm_prediction > 0 and dqn_recommendation != 0.5 and pca_components is not None:
            # Full calculation with all advanced methods
            print("[INFO] Using full Sigma calculation with all advanced models")
            sigma = (
                    0.15 * components["traditional_volatility"] +
                    0.10 * components["rsi"] +
                    0.10 * components["macd"] +
                    0.10 * components["sma_trend"] +
                    0.05 * components["bb_position"] +
                    0.05 * components["volume"] +
                    0.10 * components["momentum"] +
                    0.05 * components["vol_trend"] +
                    0.15 * components["lstm"] +
                    0.15 * components["dqn"]
            )
            # If we have PCA results, use them to make a small adjustment
            if pca_components is not None and len(pca_components) >= 2:
                # Use first two principal components to slightly adjust sigma
                # This can help capture patterns not directly modeled elsewhere
                pca_influence = np.tanh(np.sum(pca_components[:2]) / 2) * 0.05  # Small adjustment, max 5%
                sigma += pca_influence
                print(f"[DEBUG] PCA adjustment to Sigma: {pca_influence}")
                
        elif lstm_prediction > 0 and dqn_recommendation != 0.5:
            # Advanced calculation without PCA
            print("[INFO] Using advanced Sigma calculation with LSTM and DQN")
            sigma = (
                    0.15 * components["traditional_volatility"] +
                    0.10 * components["rsi"] +
                    0.10 * components["macd"] +
                    0.10 * components["sma_trend"] +
                    0.05 * components["bb_position"] +
                    0.05 * components["volume"] +
                    0.10 * components["momentum"] +
                    0.05 * components["vol_trend"] +
                    0.15 * components["lstm"] +
                    0.15 * components["dqn"]
            )
        elif lstm_prediction > 0:
            # Intermediate calculation with LSTM but no DQN
            print("[INFO] Using intermediate Sigma calculation with LSTM")
            sigma = (
                    0.20 * components["traditional_volatility"] +
                    0.15 * components["rsi"] +
                    0.15 * components["macd"] +
                    0.10 * components["sma_trend"] +
                    0.05 * components["bb_position"] +
                    0.05 * components["volume"] +
                    0.10 * components["momentum"] +
                    0.05 * components["vol_trend"] +
                    0.15 * components["lstm"]
            )
        elif dqn_recommendation != 0.5:
            # Intermediate calculation with DQN but no LSTM
            print("[INFO] Using intermediate Sigma calculation with DQN")
            sigma = (
                    0.20 * components["traditional_volatility"] +
                    0.15 * components["rsi"] +
                    0.15 * components["macd"] +
                    0.10 * components["sma_trend"] +
                    0.05 * components["bb_position"] +
                    0.05 * components["volume"] +
                    0.10 * components["momentum"] +
                    0.05 * components["vol_trend"] +
                    0.15 * components["dqn"]
            )
        else:
            # Basic calculation with traditional indicators only
            print("[INFO] Using basic Sigma calculation with traditional indicators only")
            sigma = (
                    0.25 * components["traditional_volatility"] +
                    0.20 * components["rsi"] +
                    0.15 * components["macd"] +
                    0.15 * components["sma_trend"] +
                    0.10 * components["bb_position"] +
                    0.05 * components["volume"] +
                    0.10 * components["momentum"]
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
            "momentum": momentum_signal,
            "volatility_trend": vol_trend_signal,
            "pca_variance": pca_variance if pca_variance is not None else [],
            "last_price": latest['4. close'] if not np.isnan(latest['4. close']) else 0,
            "simplified": lstm_prediction == 0 and dqn_recommendation == 0.5
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
        batch = stocks[i:i+batch_size]
        print(f"[INFO] Processing batch {i//batch_size + 1}/{(len(stocks) + batch_size - 1)//batch_size}")
        
        # Check overall timeout
        if time.time() - overall_start_time > MAX_TOTAL_RUNTIME:
            print(f"[WARNING] Maximum total runtime ({MAX_TOTAL_RUNTIME/3600:.1f} hours) reached. Stopping.")
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
            print(f"[INFO] Progress: {progress:.1f}% ({total_analyzed}/{len(stocks)}), Success rate: {success_rate:.1f}%")
        
        # Disconnect after batch is complete
        client.disconnect()
        
        # Wait between batches to avoid rate limits
        if i + batch_size < len(stocks):  # If not the last batch
            print(f"[INFO] Batch complete. Waiting {BATCH_DELAY} seconds before next batch...")
            time.sleep(BATCH_DELAY)
    
    # Final report
    print(f"[INFO] Analysis complete! Analyzed {total_analyzed}/{len(stocks)} stocks with {total_successful} successful analyses.")
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
