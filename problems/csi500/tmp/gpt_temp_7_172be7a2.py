import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Ensure we have enough data for calculations
    if len(data) < 6:
        return result
    
    # Calculate basic metrics
    data['prev_close'] = data['close'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    
    # Calculate daily ranges
    data['daily_range'] = data['high'] - data['low']
    data['prev_range'] = data['daily_range'].shift(1)
    
    # Calculate 5-day rolling averages
    data['range_5d_avg'] = data['daily_range'].rolling(window=5, min_periods=1).mean()
    data['volatility_5d_avg'] = data['daily_range'].rolling(window=5, min_periods=1).mean()
    
    # Calculate 3-day momentum
    data['close_3d_ago'] = data['close'].shift(3)
    
    for i in range(5, len(data)):
        current_data = data.iloc[i]
        prev_data = data.iloc[i-1] if i > 0 else None
        
        # Skip if missing required data
        if (pd.isna(current_data['prev_close']) or pd.isna(current_data['prev_volume']) or 
            pd.isna(current_data['prev_amount']) or pd.isna(current_data['prev_range'])):
            continue
        
        # Factor 1: Gap-Range Efficiency with Volume Confirmation
        opening_gap = (current_data['open'] - current_data['prev_close']) / current_data['prev_close']
        gap_direction = np.sign(opening_gap)
        gap_persistence = (current_data['close'] - current_data['open']) / (current_data['high'] - current_data['low'])
        range_efficiency = (current_data['close'] - current_data['open']) / (current_data['high'] - current_data['low'])
        efficiency_divergence = gap_direction * range_efficiency
        volume_intensity = current_data['volume'] / current_data['prev_volume']
        amount_concentration = current_data['amount'] / current_data['volume']
        factor1 = efficiency_divergence * volume_intensity * amount_concentration
        
        # Factor 2: Amount-Weighted Range Breakout
        current_range = current_data['high'] - current_data['low']
        range_momentum = current_range / current_data['prev_range']
        range_breakout = current_range / current_data['range_5d_avg']
        amount_to_volume = current_data['amount'] / current_data['volume']
        trade_size_momentum = current_data['amount'] / current_data['prev_amount']
        microstructure_signal = amount_to_volume * trade_size_momentum
        price_position = (current_data['close'] - current_data['low']) / (current_data['high'] - current_data['low'])
        base_breakout = range_breakout * price_position
        factor2 = base_breakout * microstructure_signal
        
        # Factor 3: Volatility-Regime Reversal with Exhaustion
        price_extremity = (current_data['close'] - current_data['low']) / (current_data['high'] - current_data['low'])
        reversal_signal = 1 - 2 * abs(price_extremity - 0.5)
        intraday_efficiency = abs(current_data['close'] - current_data['open']) / (current_data['high'] - current_data['low'])
        daily_volatility = current_data['high'] - current_data['low']
        volatility_cluster = daily_volatility / current_data['volatility_5d_avg']
        regime_adjusted_reversal = reversal_signal * volatility_cluster
        volume_cluster = current_data['volume'] / current_data['prev_volume']
        exhaustion_score = 1 - intraday_efficiency
        factor3 = regime_adjusted_reversal * exhaustion_score * volume_cluster
        
        # Factor 4: Multi-Timeframe Pressure Convergence
        short_term_pressure = (current_data['close'] - current_data['low']) / (current_data['high'] - current_data['low'])
        medium_term_momentum = current_data['close'] / current_data['close_3d_ago'] - 1 if not pd.isna(current_data['close_3d_ago']) else 0
        short_term_return = current_data['close'] / current_data['prev_close'] - 1
        direction_consistency = np.sign(short_term_return) * np.sign(medium_term_momentum)
        base_pressure = short_term_pressure * direction_consistency
        volume_confirmation = current_data['volume'] / current_data['prev_volume']
        factor4 = base_pressure * medium_term_momentum * volume_confirmation
        
        # Factor 5: Price-Efficiency with Amount Flow
        price_oscillation = (current_data['close'] - current_data['open']) / (current_data['high'] - current_data['low'])
        range_utilization = abs(current_data['close'] - current_data['open']) / (current_data['high'] - current_data['low'])
        amount_efficiency = current_data['amount'] / current_data['volume']
        efficiency_amount_alignment = price_oscillation * amount_efficiency
        overbought_oversold = 1 - 2 * abs((current_data['close'] - current_data['low']) / (current_data['high'] - current_data['low']) - 0.5)
        factor5 = efficiency_amount_alignment * overbought_oversold
        
        # Factor 6: Volume-Pressure Regime Detection
        intraday_pressure = (current_data['close'] - current_data['low']) / (current_data['high'] - current_data['low'])
        volume_intensity_v6 = current_data['volume'] / current_data['prev_volume']
        amount_concentration_v6 = current_data['amount'] / current_data['volume']
        pressure_volume_alignment = intraday_pressure * volume_intensity_v6
        large_trade_confirmation = pressure_volume_alignment * amount_concentration_v6
        factor6 = large_trade_confirmation * intraday_pressure
        
        # Factor 7: Opening-Closing Efficiency Divergence
        opening_gap_v7 = (current_data['open'] - current_data['prev_close']) / current_data['prev_close']
        # Using last hour approximation (last 25% of trading range)
        trading_hours = 4  # Assuming 4 trading hours
        last_hour_idx = max(0, i - int(trading_hours * 0.25))
        last_hour_data = data.iloc[last_hour_idx:i+1]
        
        if len(last_hour_data) > 0:
            last_hour_open = last_hour_data['open'].iloc[0]
            last_hour_high = last_hour_data['high'].max()
            last_hour_low = last_hour_data['low'].min()
            closing_efficiency = (current_data['close'] - last_hour_open) / (last_hour_high - last_hour_low) if (last_hour_high - last_hour_low) > 0 else 0
        else:
            closing_efficiency = 0
        
        # Using first hour approximation (first 25% of trading range)
        first_hour_idx = min(len(data)-1, i + int(trading_hours * 0.25))
        first_hour_data = data.iloc[i:first_hour_idx+1]
        
        if len(first_hour_data) > 0:
            first_hour_volume = first_hour_data['volume'].sum()
            first_hour_amount = first_hour_data['amount'].sum()
            opening_volume_ratio = first_hour_volume / current_data['volume'] if current_data['volume'] > 0 else 0
            opening_amount_ratio = first_hour_amount / current_data['amount'] if current_data['amount'] > 0 else 0
        else:
            opening_volume_ratio = 0
            opening_amount_ratio = 0
        
        amount_concentration_v7 = opening_amount_ratio / opening_volume_ratio if opening_volume_ratio > 0 else 0
        flow_efficiency = opening_gap_v7 * amount_concentration_v7
        session_divergence = opening_gap_v7 * closing_efficiency
        base_signal = flow_efficiency * session_divergence
        factor7 = base_signal * opening_volume_ratio
        
        # Combine all factors (equal weighting)
        combined_factor = (factor1 + factor2 + factor3 + factor4 + factor5 + factor6 + factor7) / 7
        
        result.iloc[i] = combined_factor
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
