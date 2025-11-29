import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Calculate novel microstructure alpha factors using intraday patterns, price memory,
    order flow dynamics, volatility regimes, and liquidity microstructure.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor storage
    factors = pd.Series(index=data.index, dtype=float)
    
    # Calculate basic rolling statistics
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['price_range'] = data['high'] - data['low']
    data['prev_range'] = data['prev_high'] - data['prev_low']
    
    # Rolling windows for volume and volatility
    vol_window = 20
    data['vol_avg'] = data['volume'].rolling(window=vol_window, min_periods=10).mean()
    data['range_avg'] = data['price_range'].rolling(window=vol_window, min_periods=10).mean()
    
    for i, (idx, row) in enumerate(data.iterrows()):
        if i < max(vol_window, 5):  # Ensure sufficient history
            factors.loc[idx] = 0
            continue
            
        current_data = data.iloc[:i+1]
        current_row = current_data.iloc[-1]
        
        # Skip if missing required previous data
        if pd.isna(current_row['prev_close']) or pd.isna(current_row['vol_avg']):
            factors.loc[idx] = 0
            continue
            
        # 1. Intraday Reversal Patterns
        # Opening Rejection
        if current_row['high'] != current_row['open']:
            opening_rejection = (current_row['open'] - current_row['low']) / (current_row['high'] - current_row['open'])
        else:
            opening_rejection = 0
            
        # Gap Fill Strength
        if abs(current_row['open'] - current_row['prev_close']) > 0:
            gap_fill = abs(current_row['close'] - current_row['prev_close']) / abs(current_row['open'] - current_row['prev_close'])
        else:
            gap_fill = 0
            
        # Closing Momentum
        if current_row['high'] != current_row['low']:
            closing_momentum = (current_row['close'] - current_row['low']) / (current_row['high'] - current_row['low'])
        else:
            closing_momentum = 0
            
        intraday_factor = opening_rejection + gap_fill + closing_momentum
        
        # 2. Price Level Memory
        # Previous Close Attraction
        if current_row['price_range'] > 0:
            close_attraction = abs(current_row['close'] - current_row['prev_close']) / current_row['price_range']
        else:
            close_attraction = 0
            
        # High/Low Memory
        high_memory = current_row['high'] / current_row['prev_high'] if current_row['prev_high'] > 0 else 1
        low_memory = current_row['low'] / current_row['prev_low'] if current_row['prev_low'] > 0 else 1
        
        # Price Consolidation (simplified as days since range expansion)
        recent_ranges = current_data['price_range'].tail(5)
        if len(recent_ranges) >= 5:
            consolidation = np.std(recent_ranges) / np.mean(recent_ranges) if np.mean(recent_ranges) > 0 else 0
        else:
            consolidation = 0
            
        price_memory_factor = close_attraction + (high_memory + low_memory) / 2 - consolidation
        
        # 3. Order Flow Dynamics
        # Volume Reversal Confirmation
        if current_row['vol_avg'] > 0:
            volume_reversal = current_row['volume'] / current_row['vol_avg']
        else:
            volume_reversal = 1
            
        # Bid-Ask Pressure (approximated by intraday volatility relative to volume)
        if current_row['volume'] > 0:
            spread_pressure = current_row['price_range'] / current_row['volume']
        else:
            spread_pressure = 0
            
        # Large Order Absorption (volume concentration at extremes)
        if current_row['range_avg'] > 0:
            large_order = (current_row['price_range'] / current_row['range_avg']) * volume_reversal
        else:
            large_order = 0
            
        order_flow_factor = volume_reversal + spread_pressure + large_order
        
        # 4. Volatility Regime Effects
        # Volatility Compression
        if current_row['prev_range'] > 0:
            vol_compression = current_row['price_range'] / current_row['prev_range']
        else:
            vol_compression = 1
            
        # Regime Transition (volume during volatility changes)
        regime_transition = volume_reversal * vol_compression
        
        # Mean Reversion Strength
        mean_price = (current_row['high'] + current_row['low']) / 2
        mean_reversion = abs(mean_price - current_row['close']) / current_row['price_range'] if current_row['price_range'] > 0 else 0
        
        volatility_factor = vol_compression + regime_transition + mean_reversion
        
        # 5. Liquidity Microstructure
        # Temporary Liquidity Gaps (volume dry-ups)
        recent_volumes = current_data['volume'].tail(5)
        if len(recent_volumes) >= 5:
            liquidity_gap = np.std(recent_volumes) / np.mean(recent_volumes) if np.mean(recent_volumes) > 0 else 0
        else:
            liquidity_gap = 0
            
        # Quote Update Frequency (approximated by price changes)
        recent_closes = current_data['close'].tail(3)
        if len(recent_closes) >= 3:
            quote_frequency = np.std(np.diff(recent_closes)) / np.mean(np.abs(np.diff(recent_closes))) if len(np.diff(recent_closes)) > 0 and np.mean(np.abs(np.diff(recent_closes))) > 0 else 0
        else:
            quote_frequency = 0
            
        # Cross-Timeframe Alignment (intraday vs daily consistency)
        intraday_trend = closing_momentum - 0.5  # Centered around 0.5
        daily_trend = (current_row['close'] - current_row['prev_close']) / current_row['prev_close'] if current_row['prev_close'] > 0 else 0
        timeframe_alignment = abs(intraday_trend - daily_trend)
        
        liquidity_factor = liquidity_gap + quote_frequency - timeframe_alignment
        
        # Combine all factors with equal weighting
        combined_factor = (
            intraday_factor + 
            price_memory_factor + 
            order_flow_factor + 
            volatility_factor + 
            liquidity_factor
        )
        
        factors.loc[idx] = combined_factor
    
    # Normalize the final factor
    if len(factors) > vol_window:
        factors = (factors - factors.rolling(window=vol_window, min_periods=10).mean()) / factors.rolling(window=vol_window, min_periods=10).std()
    
    return factors.fillna(0)
