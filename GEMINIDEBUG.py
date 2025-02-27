import os
import json
import numpy as np
import pandas as pd
import websocket
import time
import traceback
from threading import Thread
import signal

# XTB API credentials
XTB_USER_ID = "50540163"
XTB_PASSWORD = "Jphost2005"
XTB_WS_URL = "wss://ws.xtb.com/real"

# Set a global timeout flag
TIMEOUT_SECONDS = 120
operation_running = False
operation_timed_out = False


class TimeoutException(Exception):
    pass


# Setup timeout handler
def timeout_handler(signum, frame):
    global operation_timed_out
    operation_timed_out = True
    print("\n[TIMEOUT] Operation is taking too long and has been interrupted")
    raise TimeoutException("Operation took too long")


# WebSocket connection manager
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
        self.reconnect_count = 0
        self.login()

    def on_message(self, ws, message):
        try:
            data = json.loads(message)

            # Handle login response
            if "streamSessionId" in data:
                self.logged_in = True
                print("[INFO] Logged in successfully.")
                if not self.heartbeat_thread or not self.heartbeat_thread.is_alive():
                    self.start_heartbeat()

            # Command responses
            elif "status" in data and "returnData" in data:
                if hasattr(self, 'last_command') and self.last_command:
                    self.response_data[self.last_command] = data["returnData"]
                    print(f"[DEBUG] Stored response for command: {self.last_command}")
                    self.last_command = None
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

        if self.running and self.reconnect_count < self.max_reconnects:
            self.reconnect_count += 1
            backoff_time = min(30, 2 ** self.reconnect_count)
            print(
                f"[INFO] Attempting to reconnect in {backoff_time} seconds... (Attempt {self.reconnect_count}/{self.max_reconnects})")
            time.sleep(backoff_time)
            self.connect()
        elif self.reconnect_count >= self.max_reconnects:
            print(f"[ERROR] Maximum reconnection attempts ({self.max_reconnects}) reached. Giving up.")

    def connect(self):
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

            websocket_thread = Thread(target=self.ws.run_forever)
            websocket_thread.daemon = True
            websocket_thread.start()

            timeout = time.time() + 15
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
        def heartbeat_worker():
            print("[INFO] Starting heartbeat service")
            heartbeat_interval = 30
            while self.running and self.logged_in:
                try:
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

                time.sleep(heartbeat_interval)

            print("[INFO] Heartbeat service stopped")

        self.heartbeat_thread = Thread(target=heartbeat_worker)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()

    def send_command(self, command, arguments=None):
        if not self.logged_in and command != "login":
            print("[ERROR] Not logged in yet.")
            return None

        max_retries = 3
        for attempt in range(max_retries):
            try:
                payload = {"command": command}
                if arguments:
                    payload["arguments"] = arguments

                self.response_data[command] = None
                self.last_command = command

                payload_str = json.dumps(payload)
                print(f"[DEBUG] Sending: {payload_str[:100]}")

                if not self.ws or not self.ws.sock or not self.ws.sock.connected:
                    print("[ERROR] WebSocket not connected")
                    self.connect()
                    if not self.logged_in:
                        return None

                self.ws.send(payload_str)

                timeout = time.time() + 30
                while self.response_data[command] is None and time.time() < timeout:
                    time.sleep(0.1)

                if self.response_data[command] is None:
                    print(
                        f"[WARNING] Timeout waiting for response to command: {command}, attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))
                        continue
                else:
                    return self.response_data.get(command)

            except Exception as e:
                print(f"[ERROR] Error sending command {command}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))

        return None

    def login(self):
        login_cmd = {
            "command": "login",
            "arguments": {"userId": XTB_USER_ID, "password": XTB_PASSWORD}
        }
        print("[DEBUG] Sending login command")
        self.last_command = "login"

        try:
            self.ws.send(json.dumps(login_cmd))
        except Exception as e:
            print(f"[ERROR] Failed to send login command: {e}")

    def disconnect(self):
        self.running = False

        if self.logged_in:
            try:
                logout_cmd = {"command": "logout"}
                self.ws.send(json.dumps(logout_cmd))
                time.sleep(1)
            except:
                pass

        if self.ws:
            try:
                self.ws.close()
            except:
                pass

        print("[INFO] Disconnected from XTB")


# Function to fetch all listed stocks
def get_all_stocks(client):
    print("[INFO] Fetching all listed stocks from XTB...")
    response = client.send_command("getAllSymbols", {})

    if response is None:
        print("[ERROR] Failed to fetch stock list.")
        return []

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
        traceback.print_exc()
        return []


# Function to check if a stock exists in the XTB database
def check_stock_exists(client, ticker):
    stock_list = get_all_stocks(client)

    if not stock_list:
        return False

    # Check for exact match
    if ticker in stock_list:
        return ticker

    # Check for partial matches
    partial_matches = [stock for stock in stock_list if ticker in stock]

    if partial_matches:
        print(f"[INFO] Found {len(partial_matches)} potential matches for '{ticker}':")
        for i, match in enumerate(partial_matches[:10], 1):
            print(f"  {i}. {match}")

        if len(partial_matches) > 10:
            print(f"  ... and {len(partial_matches) - 10} more")

        try:
            selection = input("Enter the number of the correct ticker (or 0 to cancel): ")
            selection = int(selection)

            if 1 <= selection <= len(partial_matches):
                return partial_matches[selection - 1]
            else:
                print("[INFO] Selection cancelled or invalid")
                return False
        except ValueError:
            print("[ERROR] Invalid input")
            return False

    print(f"[INFO] No matches found for '{ticker}'")
    return False


# Function to fetch historical stock data
def get_stock_data(client, symbol):
    print(f"[INFO] Fetching historical data for: {symbol}")
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
            df["high"] = df["open"] + df["high"]
            df["low"] = df["open"] + df["low"]
            df["volume"] = df["vol"]
            print(f"[DEBUG] Processed data for {symbol}: {len(df)} records")
            return df[["open", "high", "low", "4. close", "volume"]]
        else:
            print(f"[WARNING] Missing required columns in {symbol} data")
            return None
    except Exception as e:
        print(f"[ERROR] Error processing data for {symbol}: {e}")
        traceback.print_exc()
        return None


# Calculate basic technical indicators
def calculate_basic_indicators(data):
    try:
        print(f"[INFO] Calculating technical indicators...")
        df = data.copy()

        # Calculate returns
        df['returns'] = df['4. close'].pct_change().fillna(0)

        # Calculate volatility (20-day rolling standard deviation)
        df['volatility'] = df['returns'].rolling(window=20).std().fillna(0)

        # Calculate Simple Moving Averages
        df['SMA20'] = df['4. close'].rolling(window=20).mean().fillna(0)
        df['SMA50'] = df['4. close'].rolling(window=50).mean().fillna(0)
        df['SMA200'] = df['4. close'].rolling(window=200).mean().fillna(0)

        # Calculate RSI
        delta = df['4. close'].diff().fillna(0)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().fillna(0)
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().fillna(0)
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs)).fillna(50)

        # Calculate MACD
        df['EMA12'] = df['4. close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['4. close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Volume analysis
        df['volume_change'] = df['volume'].pct_change().fillna(0)
        df['avg_volume_20d'] = df['volume'].rolling(window=20).mean().fillna(0)
        df['volume_ratio'] = df['volume'] / df['avg_volume_20d']

        print(f"[INFO] Technical indicators calculated successfully")
        return df
    except Exception as e:
        print(f"[ERROR] Error calculating technical indicators: {e}")
        traceback.print_exc()
        return None


# Simple trading signals calculation
def calculate_trading_signals(data):
    try:
        print(f"[INFO] Calculating trading signals...")
        df = data.copy()
        latest = df.iloc[-1]

        # Price momentum (last 5 days)
        recent_returns = df['returns'].iloc[-5:].mean()

        # Short-term trend
        short_trend = latest['SMA20'] / latest['SMA50'] - 1

        # Long-term trend
        long_trend = latest['SMA50'] / latest['SMA200'] - 1

        # RSI signal (0-1 scale, higher means oversold/bullish)
        rsi_signal = (max(0, min(100, latest['RSI'])) - 30) / 70

        # MACD signal (-1 to 1 scale)
        macd_signal = np.tanh(latest['MACD'] * 10)

        # Volume signal
        volume_signal = min(1, max(0, (latest['volume_ratio'] - 0.5) / 2))

        # Volatility as percentage
        volatility = latest['volatility'] * 100

        # Calculate final score (0-100 scale)
        score = (
                20 * (recent_returns * 100 + 0.5) +  # Recent returns
                20 * (short_trend + 0.5) +  # Short-term trend
                15 * (long_trend + 0.5) +  # Long-term trend
                15 * rsi_signal +  # RSI
                20 * (macd_signal + 1) / 2 +  # MACD
                10 * volume_signal  # Volume
        )

        # Ensure score is between 0 and 100
        score = max(0, min(100, score))

        # For detailed analysis, return a dictionary
        analysis = {
            "score": score,
            "recent_returns": recent_returns * 100,  # convert to percentage
            "short_term_trend": short_trend * 100,  # convert to percentage
            "long_term_trend": long_trend * 100,  # convert to percentage
            "rsi": latest['RSI'],
            "macd": latest['MACD'],
            "macd_signal": latest['MACD_signal'],
            "volume_ratio": latest['volume_ratio'],
            "volatility": volatility,
            "last_price": latest['4. close'],
            "last_date": df.index[-1].strftime('%Y-%m-%d')
        }

        print(f"[INFO] Trading signals calculated successfully")
        return analysis
    except Exception as e:
        print(f"[ERROR] Error calculating trading signals: {e}")
        traceback.print_exc()
        return None


# Function to get stock recommendation
def get_recommendation(score):
    if score > 80:
        return "STRONG BUY - Multiple technical indicators showing positive signals"
    elif score > 60:
        return "BUY - Positive trend with favorable indicators"
    elif score > 40:
        return "HOLD - Mixed signals with no clear direction"
    elif score > 20:
        return "SELL - Negative trend with bearish indicators"
    else:
        return "STRONG SELL - Multiple technical indicators showing negative signals"


# Save analysis results to text file
def save_to_text_file(stock, analysis, recommendation, file_path=None):
    if file_path is None:
        file_path = os.path.expanduser("~/Desktop/stock_analysis.txt")

    try:
        with open(file_path, "a") as file:
            file.write("\n===== STOCK ANALYSIS REPORT =====\n")
            file.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            file.write(f"SYMBOL: {stock}\n")
            file.write(f"LAST PRICE: ${analysis['last_price']:.2f} (as of {analysis['last_date']})\n")
            file.write(f"SCORE: {analysis['score']:.1f}/100\n")
            file.write(f"RECOMMENDATION: {recommendation}\n\n")

            file.write("TECHNICAL ANALYSIS:\n")
            file.write(f"Recent Returns (5 days): {analysis['recent_returns']:.2f}%\n")
            file.write(f"Short-term Trend: {analysis['short_term_trend']:.2f}%\n")
            file.write(f"Long-term Trend: {analysis['long_term_trend']:.2f}%\n")
            file.write(f"RSI (14-day): {analysis['rsi']:.1f}\n")
            file.write(f"MACD: {analysis['macd']:.4f}\n")
            file.write(f"MACD Signal: {analysis['macd_signal']:.4f}\n")
            file.write(f"Volume Ratio (vs 20-day avg): {analysis['volume_ratio']:.2f}x\n")
            file.write(f"Volatility (20-day): {analysis['volatility']:.2f}%\n")
            file.write("--------------------------------------\n")

        print(f"[INFO] Analysis saved to {file_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Error saving to file: {e}")
        traceback.print_exc()
        return False


# Main function to analyze a single stock
def analyze_single_stock(ticker):
    global operation_running, operation_timed_out

    print(f"[INFO] Starting analysis for ticker: {ticker}")

    # Set timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    # Connect to XTB
    client = XTBClient()
    operation_running = True

    try:
        if not client.connect():
            print("[ERROR] Failed to connect to XTB. Exiting.")
            return False

        # Check if the stock exists
        stock_symbol = check_stock_exists(client, ticker)

        if not stock_symbol:
            print(f"[ERROR] Could not find ticker {ticker} in XTB database")
            client.disconnect()
            return False

        # If stock_symbol is a string returned from check_stock_exists, use it
        if isinstance(stock_symbol, str):
            ticker = stock_symbol

        print(f"[INFO] Analyzing {ticker}...")

        # Get stock data
        data = get_stock_data(client, ticker)

        if data is None or len(data) < 50:  # Need at least 50 days of data
            print(f"[ERROR] Insufficient historical data for {ticker}")
            client.disconnect()
            return False

        # Calculate basic indicators
        indicators_df = calculate_basic_indicators(data)

        if indicators_df is None:
            print(f"[ERROR] Failed to calculate indicators for {ticker}")
            client.disconnect()
            return False

        # Calculate trading signals
        analysis = calculate_trading_signals(indicators_df)

        if analysis is None:
            print(f"[ERROR] Failed to calculate trading signals for {ticker}")
            client.disconnect()
            return False

        # Get recommendation
        score = analysis["score"]
        recommendation = get_recommendation(score)

        # Print results
        print("\n===== ANALYSIS RESULTS =====")
        print(f"TICKER: {ticker}")
        print(f"PRICE: ${analysis['last_price']:.2f}")
        print(f"SCORE: {score:.1f}/100")
        print(f"RECOMMENDATION: {recommendation}")
        print("===========================\n")

        # Save to file
        save_to_text_file(ticker, analysis, recommendation)

        # Cancel the timeout alarm
        signal.alarm(0)
        operation_running = False

        return True

    except TimeoutException:
        print(f"[ERROR] Analysis timed out after {TIMEOUT_SECONDS} seconds")
        return False
    except KeyboardInterrupt:
        print("\n[INFO] Analysis interrupted by user")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error during analysis: {e}")
        traceback.print_exc()
        return False
    finally:
        # Reset the alarm
        signal.alarm(0)
        operation_running = False

        # Disconnect from XTB
        if client:
            client.disconnect()


# Main execution
if __name__ == "__main__":
    print("[INFO] Stock Analyzer - Simple Version")
    print("====================================")

    try:
        # Get ticker from user
        ticker = input("Enter stock ticker to analyze (e.g., AAPL, MSFT): ").strip().upper()

        if not ticker:
            print("[ERROR] No ticker provided. Exiting.")
            exit(1)

        # Run analysis
        success = analyze_single_stock(ticker)

        if success:
            print("[INFO] Analysis completed successfully.")
            print("[INFO] Results saved to Desktop/stock_analysis.txt")
        else:
            if operation_timed_out:
                print("[ERROR] Analysis failed due to timeout.")
            else:
                print("[ERROR] Analysis failed.")

    except KeyboardInterrupt:
        print("\n[INFO] Analysis interrupted by user.")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        traceback.print_exc()