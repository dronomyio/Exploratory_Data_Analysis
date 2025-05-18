#!/usr/bin/env python3
"""
Script to fetch Ford stock data for 2009-2013 period using Yahoo Finance API
"""

import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Convert dates to epoch timestamps
def date_to_epoch(date_str):
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return int(time.mktime(dt.timetuple()))

# Start date: January 1, 2009
start_date = date_to_epoch('2009-01-01')
# End date: December 31, 2013
end_date = date_to_epoch('2013-12-31')

# Initialize API client
client = ApiClient()

# Fetch Ford stock data for 2009-2013
print("Fetching Ford stock data for 2009-2013...")
ford_data = client.call_api('YahooFinance/get_stock_chart', query={
    'symbol': 'F',
    'region': 'US',
    'interval': '1d',  # Daily data
    'period1': str(start_date),
    'period2': str(end_date),
    'includeAdjustedClose': 'True'
})

# Process the data
if 'chart' in ford_data and 'result' in ford_data['chart'] and ford_data['chart']['result']:
    result = ford_data['chart']['result'][0]
    
    # Extract timestamps and convert to dates
    timestamps = result['timestamp']
    dates = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps]
    
    # Extract price data
    indicators = result['indicators']
    
    # Get adjusted close prices
    adj_close = indicators['adjclose'][0]['adjclose']
    
    # Get regular prices
    quote = indicators['quote'][0]
    opens = quote['open']
    highs = quote['high']
    lows = quote['low']
    closes = quote['close']
    volumes = quote['volume']
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Adj Close': adj_close,
        'Volume': volumes
    })
    
    # Calculate returns (percentage change in adjusted close)
    df['Return'] = df['Adj Close'].pct_change()
    
    # Drop the first row (NaN return)
    df = df.dropna()
    
    # Save to CSV
    df.to_csv('RecentFord.csv', index=False)
    print(f"Data saved to RecentFord.csv with {len(df)} rows")
    
    # Find the largest negative return
    min_return_idx = df['Return'].idxmin()
    min_return_date = df.loc[min_return_idx, 'Date']
    min_return_value = df.loc[min_return_idx, 'Return']
    
    print(f"Largest negative return: {min_return_value:.6f} on {min_return_date}")
    
else:
    print("Error fetching data:", ford_data)
