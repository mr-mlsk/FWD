import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import warnings
import time
import logging
import json
import os
import sys
from typing import Optional, Dict, Any
import signal
import threading
import traceback

warnings.filterwarnings('ignore')

class LiveTradingBot:
    def __init__(self, api_key: str, api_secret: str, config_file: str = 'config.json'):
        """
        Live Trading Bot for Nadaraya-Watson Strategy
        
        Parameters:
        - api_key: Binance API key
        - api_secret: Binance API secret
        - config_file: Configuration file path
        """
        
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Initialize Binance client
        self.client = Client(api_key, api_secret)
        
        # Strategy parameters (optimized values)
        self.symbol = self.config.get('symbol', 'BTCUSDT')
        self.interval = self.config.get('interval', '15m')  # Changed to 15m
        self.envelope_lookback = self.config.get('envelope_lookback', 200)
        self.h = self.config.get('h', 12.0)
        self.mult = self.config.get('mult', 2.0)
        self.atr_mult = self.config.get('atr_mult', 1.5)
        self.tp_mult = self.config.get('tp_mult', 2.5)
        self.leverage = self.config.get('leverage', 5.0)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)
        
        # Trading state
        self.position = None
        self.capital = self.config.get('initial_capital', 1000.0)
        self.is_running = False
        self.price_data = pd.DataFrame()
        
        # Setup logging
        self.setup_logging()
        
        # Validate API connection
        self.validate_api()
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Create default config with 15m interval and email alerts
            default_config = {
                "symbol": "BTCUSDT",
                "interval": "15m",
                "envelope_lookback": 200,
                "h": 12.0,
                "mult": 2.0,
                "atr_mult": 1.5,
                "tp_mult": 2.5,
                "leverage": 5.0,
                "risk_per_trade": 0.02,
                "initial_capital": 1000.0,
                "testnet": True,
                "log_level": "INFO",
                "max_positions": 1,
                "enable_email_alerts": True,
                "email_config": {
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "email": "probotroge@gmail.com",
                    "password": "your_app_password_here"
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            
            print(f"Created default config file: {config_file}")
            print("⚠️ Please update the email password in config.json for alerts to work")
            return default_config
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('NWTradingBot')
        self.logger.info("Trading bot initialized with 15m timeframe")
    
    def validate_api(self):
        """Validate Binance API connection"""
        try:
            account_info = self.client.get_account()
            self.logger.info("✓ Successfully connected to Binance API")
            
            # Check if testnet
            if self.config.get('testnet', True):
                self.logger.info("🧪 Running in TESTNET mode")
            else:
                self.logger.warning("⚠️ Running in LIVE trading mode")
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Binance API: {str(e)}")
            raise
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.is_running = False
    
    def get_historical_data(self, limit: int = 300) -> pd.DataFrame:
        """Fetch recent historical data for strategy calculation"""
        try:
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    def gaussian_kernel(self, x: np.ndarray, h: float) -> np.ndarray:
        """Gaussian kernel function"""
        return np.exp(-(x**2) / (2 * h**2))
    
    def calculate_nadaraya_watson(self, prices: np.ndarray) -> tuple:
        """Calculate Nadaraya-Watson envelope"""
        if len(prices) < self.envelope_lookback:
            return np.nan, np.nan, np.nan
        
        # Use the lookback window
        window_prices = prices[-self.envelope_lookback:]
        
        # Calculate weights (most recent gets highest weight)
        indices = np.arange(self.envelope_lookback)
        weights = self.gaussian_kernel(indices - (self.envelope_lookback-1), self.h)
        weights_sum = np.sum(weights)
        
        if weights_sum != 0:
            nw_estimate = np.sum(window_prices * weights) / weights_sum
        else:
            nw_estimate = prices[-1]
        
        # Calculate MAE
        window_estimates = np.full(len(window_prices), nw_estimate)
        mae = np.mean(np.abs(window_prices - window_estimates)) * self.mult
        
        upper_band = nw_estimate + mae
        lower_band = nw_estimate - mae
        
        return nw_estimate, upper_band, lower_band
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR and RSI"""
        df = df.copy()
        
        # ATR calculation
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        df['atr'] = tr.rolling(window=14, min_periods=1).mean()
        
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> tuple:
        """Generate trading signals"""
        if len(df) < self.envelope_lookback + 20:  # Need enough data
            return None, None
        
        # Calculate indicators
        df = self.calculate_technical_indicators(df)
        
        # Get current and previous prices
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        current_rsi = df['rsi'].iloc[-1]
        current_atr = df['atr'].iloc[-1]
        
        # Calculate envelope
        prices = df['close'].values
        nw_line, upper_band, lower_band = self.calculate_nadaraya_watson(prices)
        
        if np.isnan(upper_band) or np.isnan(lower_band):
            return None, None
        
        # Generate signals
        long_signal = (current_price < lower_band and prev_price >= lower_band and current_rsi < 30)
        short_signal = (current_price > upper_band and prev_price <= upper_band and current_rsi > 70)
        
        signal_data = {
            'current_price': current_price,
            'nw_line': nw_line,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'rsi': current_rsi,
            'atr': current_atr
        }
        
        if long_signal:
            return 'long', signal_data
        elif short_signal:
            return 'short', signal_data
        else:
            return None, signal_data
    
    def get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            account = self.client.get_account()
            
            # Find USDT balance
            for balance in account['balances']:
                if balance['asset'] == 'USDT':
                    return float(balance['free'])
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting account balance: {str(e)}")
            return 0.0
    
    def place_order(self, side: str, quantity: float, price: float = None, order_type: str = 'MARKET') -> Optional[Dict]:
        """Place an order on Binance"""
        try:
            if order_type == 'MARKET':
                if side == 'BUY':
                    order = self.client.order_market_buy(
                        symbol=self.symbol,
                        quoteOrderQty=quantity
                    )
                else:
                    # For selling, we need to specify quantity in base asset
                    # This is a simplified approach - you may need to adjust
                    order = self.client.order_market_sell(
                        symbol=self.symbol,
                        quantity=quantity
                    )
            else:
                order = self.client.create_order(
                    symbol=self.symbol,
                    side=side,
                    type=order_type,
                    quantity=quantity,
                    price=price
                )
            
            self.logger.info(f"Order placed: {order}")
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return None
    
    def calculate_position_size(self, signal_data: Dict) -> float:
        """Calculate position size based on risk management"""
        balance = self.get_account_balance()
        risk_amount = balance * self.risk_per_trade
        
        # Calculate stop loss distance
        current_price = signal_data['current_price']
        atr = signal_data['atr']
        
        if signal_data.get('signal_type') == 'long':
            stop_loss = signal_data['lower_band'] - (self.atr_mult * atr)
        else:
            stop_loss = signal_data['upper_band'] + (self.atr_mult * atr)
        
        stop_distance = abs(current_price - stop_loss) / current_price
        
        # Position size considering leverage
        position_value = risk_amount / stop_distance
        leveraged_position = position_value * self.leverage
        
        # Don't risk more than available balance
        max_position = balance * 0.95  # Keep some buffer
        final_position = min(leveraged_position, max_position)
        
        return final_position
    
    def open_position(self, signal_type: str, signal_data: Dict):
        """Open a new position"""
        if self.position is not None:
            self.logger.warning("Position already open, skipping new signal")
            return
        
        try:
            current_price = signal_data['current_price']
            atr = signal_data['atr']
            
            # Calculate position size
            position_size = self.calculate_position_size(signal_data)
            
            if position_size < 10:  # Minimum order size on Binance
                self.logger.warning(f"Position size too small: ${position_size:.2f}")
                return
            
            # Calculate stop loss and take profit
            if signal_type == 'long':
                stop_loss = signal_data['lower_band'] - (self.atr_mult * atr)
                side = 'BUY'
            else:
                stop_loss = signal_data['upper_band'] + (self.atr_mult * atr)
                side = 'SELL'
            
            stop_distance = abs(current_price - stop_loss)
            take_profit = current_price + (stop_distance * self.tp_mult) if signal_type == 'long' else current_price - (stop_distance * self.tp_mult)
            
            # Place market order
            order = self.place_order(side, position_size)
            
            if order:
                self.position = {
                    'type': signal_type,
                    'entry_time': datetime.now(),
                    'entry_price': current_price,
                    'quantity': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'order_id': order.get('orderId'),
                    'signal_data': signal_data
                }
                
                self.logger.info(f"🚀 Opened {signal_type} position:")
                self.logger.info(f"   Entry Price: ${current_price:.4f}")
                self.logger.info(f"   Position Size: ${position_size:.2f}")
                self.logger.info(f"   Stop Loss: ${stop_loss:.4f}")
                self.logger.info(f"   Take Profit: ${take_profit:.4f}")
                
                # Send alert
                alert_message = f"""
🚀 NEW POSITION OPENED - {signal_type.upper()}

Symbol: {self.symbol}
Entry Price: ${current_price:.4f}
Position Size: ${position_size:.2f}
Stop Loss: ${stop_loss:.4f}
Take Profit: ${take_profit:.4f}
RSI: {signal_data['rsi']:.1f}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Nadaraya-Watson Trading Bot - 15m Timeframe
"""
                self.send_alert(f"Position Opened - {signal_type.upper()}", alert_message)
        
        except Exception as e:
            self.logger.error(f"Error opening position: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def close_position(self, reason: str = "Manual"):
        """Close current position"""
        if self.position is None:
            return
        
        try:
            current_price = self.price_data['close'].iloc[-1] if not self.price_data.empty else 0
            
            # Place opposite order to close position
            if self.position['type'] == 'long':
                side = 'SELL'
            else:
                side = 'BUY'
            
            # For simplicity, using market order to close
            order = self.place_order(side, self.position['quantity'])
            
            if order:
                # Calculate P&L
                entry_price = self.position['entry_price']
                if self.position['type'] == 'long':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                
                pnl_dollar = self.position['quantity'] * pnl_pct
                duration = datetime.now() - self.position['entry_time']
                
                self.logger.info(f"📈 Closed {self.position['type']} position:")
                self.logger.info(f"   Exit Price: ${current_price:.4f}")
                self.logger.info(f"   P&L: {pnl_pct*100:.2f}% (${pnl_dollar:.2f})")
                self.logger.info(f"   Duration: {duration}")
                self.logger.info(f"   Reason: {reason}")
                
                # Send alert
                alert_message = f"""
📈 POSITION CLOSED - {self.position['type'].upper()}

Symbol: {self.symbol}
Entry Price: ${entry_price:.4f}
Exit Price: ${current_price:.4f}
P&L: {pnl_pct*100:.2f}% (${pnl_dollar:.2f})
Duration: {duration}
Reason: {reason}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Updated Capital: ${self.capital + pnl_dollar:.2f}

Nadaraya-Watson Trading Bot - 15m Timeframe
"""
                self.send_alert(f"Position Closed - {reason}", alert_message)
                
                # Update capital (simplified)
                self.capital += pnl_dollar
                
                self.position = None
        
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def check_position_exit(self, current_price: float):
        """Check if position should be closed"""
        if self.position is None:
            return
        
        stop_loss = self.position['stop_loss']
        take_profit = self.position['take_profit']
        
        if self.position['type'] == 'long':
            if current_price <= stop_loss:
                self.close_position("Stop Loss")
            elif current_price >= take_profit:
                self.close_position("Take Profit")
        else:  # short
            if current_price >= stop_loss:
                self.close_position("Stop Loss")
            elif current_price <= take_profit:
                self.close_position("Take Profit")
    
    def send_alert(self, subject: str, message: str):
        """Send email alert if configured"""
        if not self.config.get('enable_email_alerts', False):
            return
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            email_config = self.config.get('email_config', {})
            
            # Skip if no password configured
            if not email_config.get('password') or email_config.get('password') == 'your_app_password_here':
                self.logger.warning("Email password not configured, skipping alert")
                return
            
            msg = MIMEMultipart()
            msg['From'] = email_config['email']
            msg['To'] = email_config['email']
            msg['Subject'] = f"🤖 Nadaraya-Watson Bot: {subject}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['email'], email_config['password'])
            text = msg.as_string()
            server.sendmail(email_config['email'], email_config['email'], text)
            server.quit()
            
            self.logger.info(f"📧 Alert sent: {subject}")
            
        except Exception as e:
            self.logger.error(f"Error sending alert: {str(e)}")
    
    def save_state(self):
        """Save current bot state"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'capital': self.capital,
            'position': self.position,
            'is_running': self.is_running,
            'interval': self.interval
        }
        
        with open('bot_state.json', 'w') as f:
            json.dump(state, f, indent=4, default=str)
    
    def load_state(self):
        """Load bot state if exists"""
        try:
            if os.path.exists('bot_state.json'):
                with open('bot_state.json', 'r') as f:
                    state = json.load(f)
                
                self.capital = state.get('capital', self.capital)
                self.position = state.get('position')
                
                self.logger.info("Previous state loaded")
                
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")
    
    def send_startup_alert(self):
        """Send startup notification"""
        startup_message = f"""
🤖 TRADING BOT STARTED

Symbol: {self.symbol}
Timeframe: {self.interval}
Strategy: Nadaraya-Watson
Parameters:
- Envelope Lookback: {self.envelope_lookback}
- Bandwidth (h): {self.h}
- Multiplier: {self.mult}
- ATR Multiplier: {self.atr_mult}
- Take Profit Multiplier: {self.tp_mult}
- Leverage: {self.leverage}x
- Risk per Trade: {self.risk_per_trade*100}%

Initial Capital: ${self.capital:.2f}
Mode: {'TESTNET' if self.config.get('testnet', True) else 'LIVE TRADING'}

Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_alert("Bot Started", startup_message)
    
    def run(self):
        """Main trading loop"""
        self.logger.info("🚀 Starting live trading bot...")
        self.logger.info(f"Configuration: {self.symbol} {self.interval} timeframe")
        self.logger.info(f"Strategy: h={self.h}, mult={self.mult}, atr_mult={self.atr_mult}, tp_mult={self.tp_mult}")
        
        # Load previous state
        self.load_state()
        
        # Send startup alert
        self.send_startup_alert()
        
        self.is_running = True
        
        try:
            while self.is_running:
                try:
                    # Fetch latest data
                    self.price_data = self.get_historical_data()
                    
                    if self.price_data.empty:
                        self.logger.warning("No price data received")
                        time.sleep(60)
                        continue
                    
                    current_price = self.price_data['close'].iloc[-1]
                    
                    # Check if we need to exit existing position
                    if self.position is not None:
                        self.check_position_exit(current_price)
                    
                    # Generate signals
                    signal_type, signal_data = self.generate_signal(self.price_data)
                    
                    # Log current status
                    if signal_data:
                        self.logger.info(f"💹 Current Price: ${current_price:.4f} | "
                                       f"RSI: {signal_data['rsi']:.1f} | "
                                       f"Upper Band: ${signal_data['upper_band']:.4f} | "
                                       f"Lower Band: ${signal_data['lower_band']:.4f} | "
                                       f"Position: {'None' if self.position is None else self.position['type']}")
                    
                    # Act on signals
                    if signal_type and self.position is None:
                        self.open_position(signal_type, {**signal_data, 'signal_type': signal_type})
                    
                    # Save state
                    self.save_state()
                    
                    # Wait for next candle - 15 minutes = 900 seconds
                    interval_seconds = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400, '1d': 86400}
                    wait_time = interval_seconds.get(self.interval, 900)
                    
                    self.logger.info(f"⏰ Waiting {wait_time}s for next 15m candle...")
                    time.sleep(wait_time)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    
                    # Send error alert
                    error_message = f"""
❌ ERROR IN TRADING BOT

Error: {str(e)}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Bot will retry in 60 seconds...

Traceback:
{traceback.format_exc()}
"""
                    self.send_alert("Bot Error", error_message)
                    
                    time.sleep(60)  # Wait before retrying
                    
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            self.logger.info("🛑 Shutting down trading bot...")
            
            # Close any open positions
            if self.position is not None:
                self.close_position("Bot Shutdown")
            
            # Send shutdown alert
            shutdown_message = f"""
🛑 TRADING BOT SHUTDOWN

Final Capital: ${self.capital:.2f}
Shutdown Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Position at Shutdown: {'None' if self.position is None else self.position['type']}

Bot has been stopped gracefully.
"""
            self.send_alert("Bot Shutdown", shutdown_message)
            
            # Save final state
            self.save_state()
            
            self.logger.info("✅ Trading bot shutdown complete")


def main():
    """Main function"""
    print("🤖 Nadaraya-Watson Live Trading Bot - 15m Timeframe")
    print("=" * 60)
    
    # Load API credentials
    try:
        from api import api_setup
        api_key = api_setup["apiKey"]
        api_secret = api_setup["secret"]
        print("✓ API credentials loaded")
    except ImportError:
        print("❌ Please create 'api.py' file with your Binance API credentials:")
        print("api_setup = {'apiKey': 'your_api_key', 'secret': 'your_api_secret'}")
        return
    
    # Initialize and run bot
    try:
        bot = LiveTradingBot(api_key, api_secret)
        bot.run()
    except Exception as e:
        print(f"❌ Error starting bot: {str(e)}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
