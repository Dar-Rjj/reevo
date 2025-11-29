import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel alpha factor combining price-volume microstructure signals
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    # Required parameters
    short_window = 5
    medium_window = 10
    long_window = 20
    
    for current_date in df.index:
        current_idx = df.index.get_loc(current_date)
        if current_idx < long_window:
            factor_values.loc[current_date] = 0
            continue
            
        # Get current and historical data
        current_data = df.iloc[current_idx]
        hist_data = df.iloc[:current_idx+1]
        
        # 1. Intraday Reversal Strength
        recent_high = hist_data['high'].tail(short_window).max()
        recent_low = hist_data['low'].tail(short_window).min()
        recent_range = recent_high - recent_low
        
        if recent_range > 0:
            price_deviation = (current_data['close'] - (recent_high + recent_low) / 2) / recent_range
        else:
            price_deviation = 0
            
        # Volume acceleration (current vs recent average)
        recent_volume_avg = hist_data['volume'].tail(short_window-1).mean()
        if recent_volume_avg > 0:
            volume_accel = current_data['volume'] / recent_volume_avg - 1
        else:
            volume_accel = 0
            
        # Retracement momentum (current vs previous close momentum)
        prev_close = hist_data['close'].iloc[-2] if len(hist_data) > 1 else current_data['close']
        price_momentum = (current_data['close'] - prev_close) / prev_close
        
        reversal_strength = price_deviation * volume_accel * np.sign(-price_momentum)
        
        # 2. Range Breakout Efficiency
        medium_high = hist_data['high'].tail(medium_window).max()
        medium_low = hist_data['low'].tail(medium_window).min()
        medium_range = medium_high - medium_low
        
        # Compression detection (recent range vs medium range)
        if medium_range > 0:
            compression_ratio = recent_range / medium_range
        else:
            compression_ratio = 1
            
        # Volume decline validation
        medium_volume_avg = hist_data['volume'].tail(medium_window).mean()
        short_volume_avg = hist_data['volume'].tail(short_window).mean()
        if medium_volume_avg > 0:
            volume_decline = short_volume_avg / medium_volume_avg
        else:
            volume_decline = 1
            
        # Breakout strength
        if recent_range > 0:
            breakout_strength = (current_data['close'] - (recent_high + recent_low) / 2) / recent_range
        else:
            breakout_strength = 0
            
        breakout_efficiency = compression_ratio * volume_decline * breakout_strength
        
        # 3. Momentum Divergence Signals
        # Volume-weighted vs simple price momentum
        recent_prices = hist_data['close'].tail(medium_window)
        recent_volumes = hist_data['volume'].tail(medium_window)
        
        if len(recent_prices) > 1:
            simple_momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            
            # Volume-weighted momentum
            price_changes = recent_prices.pct_change().dropna()
            volume_weights = recent_volumes.iloc[1:] / recent_volumes.iloc[1:].sum()
            if len(price_changes) == len(volume_weights):
                vw_momentum = (price_changes * volume_weights).sum()
            else:
                vw_momentum = simple_momentum
        else:
            simple_momentum = 0
            vw_momentum = 0
            
        # Divergence magnitude
        momentum_divergence = vw_momentum - simple_momentum
        
        # Volume confirmation
        if simple_momentum != 0:
            volume_confirmation = np.sign(simple_momentum) * volume_accel
        else:
            volume_confirmation = 0
            
        momentum_signal = momentum_divergence * volume_confirmation
        
        # 4. Session Transition Dynamics
        # Opening auction efficiency (first hour vs previous close)
        if current_idx >= 1:
            prev_close = hist_data['close'].iloc[-2]
            open_gap = (current_data['open'] - prev_close) / prev_close
            current_range = (current_data['high'] - current_data['low']) / current_data['open']
            
            if current_range > 0:
                opening_efficiency = open_gap / current_range
            else:
                opening_efficiency = open_gap
        else:
            opening_efficiency = 0
            
        # Closing momentum (last hour vs daily range)
        daily_range = (current_data['high'] - current_data['low']) / current_data['open']
        if daily_range > 0:
            close_position = (current_data['close'] - current_data['low']) / (current_data['high'] - current_data['low'])
            closing_momentum = (2 * close_position - 1) * daily_range
        else:
            closing_momentum = 0
            
        session_dynamics = opening_efficiency + closing_momentum
        
        # 5. Compression-Expansion Timing
        # Price range contraction
        long_high = hist_data['high'].tail(long_window).max()
        long_low = hist_data['low'].tail(long_window).min()
        long_range = long_high - long_low
        
        if long_range > 0:
            range_contraction = recent_range / long_range
        else:
            range_contraction = 1
            
        # Volume pattern (current vs historical volatility)
        volume_std = hist_data['volume'].tail(long_window).std()
        volume_mean = hist_data['volume'].tail(long_window).mean()
        
        if volume_mean > 0:
            volume_zscore = (current_data['volume'] - volume_mean) / volume_std if volume_std > 0 else 0
        else:
            volume_zscore = 0
            
        # Expansion signals
        price_volatility = hist_data['close'].pct_change().tail(medium_window).std()
        if price_volatility > 0:
            current_volatility = abs((current_data['close'] - current_data['open']) / current_data['open'])
            expansion_signal = current_volatility / price_volatility * range_contraction
        else:
            expansion_signal = range_contraction
            
        compression_timing = range_contraction * volume_zscore * expansion_signal
        
        # Combine all components with weights
        factor_value = (
            0.25 * reversal_strength +
            0.20 * breakout_efficiency +
            0.20 * momentum_signal +
            0.20 * session_dynamics +
            0.15 * compression_timing
        )
        
        factor_values.loc[current_date] = factor_value
    
    # Final normalization
    rolling_mean = factor_values.rolling(window=medium_window, min_periods=1).mean()
    rolling_std = factor_values.rolling(window=medium_window, min_periods=1).std()
    
    normalized_factor = (factor_values - rolling_mean) / rolling_std
    normalized_factor = normalized_factor.replace([np.inf, -np.inf], 0).fillna(0)
    
    return normalized_factor
