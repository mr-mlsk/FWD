import os
import time
import csv
import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# --- STRATEGY PARAMETERS (optimized defaults) ---
SYMBOL = 'BTCUSDT'
INTERVAL = '4h'
ENVELOPE_LOOKBACK = 200
H = 12
MULT = 3.0
ATR_MULT = 1.5
TP_MULT = 2.0
LEVERAGE = 3.0
RISK_PER_TRADE = 0.02  # 2% risk per trade
INITIAL_CAPITAL = 10000

# Load API keys from environment
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
if not API_KEY or not API_SECRET:
    raise Exception('Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables')

client = Client(API_KEY, API_SECRET)

# CSV file for logging forward trades
LOG_FILE = 'forward_trades.csv'
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'signal', 'entry_price', 'stop_loss', 'take_profit',
            'risk_amount', 'size', 'status', 'exit_price', 'exit_time', 'pnl'
        ])

def fetch_historical(start, end):
    df = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=30), end)
        data = client.get_historical_klines(
            SYMBOL, INTERVAL,
            current.strftime("%d %b %Y %H:%M:%S"),
            chunk_end.strftime("%d %b %Y %H:%M:%S"),
            limit=1000
        )
        if not data:
            break
        df += data
        last_close = int(data[-1][6])
        current = datetime.utcfromtimestamp(last_close/1000 + 1)
        time.sleep(0.1)
    df = pd.DataFrame(df, columns=[
        'ts','open','high','low','close','vol','ct','qv','tr','tb','tq','ignore'
    ])
    df[['open','high','low','close','vol']] = df[['open','high','low','close','vol']].astype(float)
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    df = df[~df.index.duplicated()]
    return df

def gaussian_envelope(series):
    prices = series.values
    n = len(prices)
    est = np.full(n, np.nan)
    for i in range(ENVELOPE_LOOKBACK, n):
        w = np.exp(-((np.arange(ENVELOPE_LOOKBACK)-(ENVELOPE_LOOKBACK-1))**2)/(2*H**2))
        window = prices[i-ENVELOPE_LOOKBACK:i]
        est[i] = np.dot(window, w)/w.sum()
    errors = np.abs(prices-est)
    mae = pd.Series(errors, index=series.index).rolling(ENVELOPE_LOOKBACK).mean() * MULT
    return pd.Series(est, series.index), est+mae, est-mae

def calculate_atr(df):
    high, low, close = df['high'], df['low'], df['close']
    prev = close.shift(1)
    tr = np.maximum(high-low, np.maximum(abs(high-prev), abs(low-prev)))
    return tr.rolling(14).mean()

def generate_signal(df):
    df['nw'], df['ub'], df['lb'] = gaussian_envelope(df['close'])
    df['atr'] = calculate_atr(df)
    df.dropna(inplace=True)
    last = df.iloc[-1]
    price = last['close']
    if price < last['lb']:
        return 'LONG', price, last['lb'], last['ub'], last['atr']
    elif price > last['ub']:
        return 'SHORT', price, last['ub'], last['lb'], last['atr']
    else:
        return None, None, None, None, None

def log_trade(row):
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def run_forward_test():
    capital = INITIAL_CAPITAL
    start = datetime.utcnow() - timedelta(days=730 + ENVELOPE_LOOKBACK*2)
    while True:
        now = datetime.utcnow()
        df = fetch_historical(start, now)
        signal, entry, sl_ref, tp_ref, atr_val = generate_signal(df)
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        if signal:
            # Compute SL, TP, position sizing
            if signal == 'LONG':
                sl = sl_ref - ATR_MULT * atr_val
                tp = entry + (entry - sl) * TP_MULT
                direction = 1
            else:
                sl = sl_ref + ATR_MULT * atr_val
                tp = entry - (sl - entry) * TP_MULT
                direction = -1
            risk_amount = capital * RISK_PER_TRADE
            stop_dist = abs(entry - sl)
            size = risk_amount / (stop_dist / entry)
            # Log entry
            log_trade([
                timestamp, signal, f"{entry:.2f}", f"{sl:.2f}", f"{tp:.2f}",
                f"{risk_amount:.2f}", f"{size:.6f}", 'OPEN', '', '', ''
            ])
            print(f"{timestamp} | {signal} @ {entry:.2f} sl {sl:.2f} tp {tp:.2f} size {size:.6f}")
        else:
            print(f"{timestamp} | No signal")

        # Sleep until next interval close + small buffer
        interval_seconds = {'1h':3600, '4h':14400, '1d':86400}.get(INTERVAL,14400)
        next_sleep = interval_seconds - (time.time() % interval_seconds) + 5
        time.sleep(next_sleep)

if __name__ == "__main__":
    run_forward_test()
