import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel Intraday Gap Momentum with Volume-Weighted Volatility Components
    """
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    
    # Remove first rows with NaN values from shifts
    data = data.dropna()
    
    # Initialize factor series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    for i, (date, row) in enumerate(data.iterrows()):
        if i < 10:  # Need enough history for acceleration calculations
            factor_values[date] = 0
            continue
            
        # === Component 1: Volatility-Adjusted Gap Momentum with Volume Confirmation ===
        # Raw Opening Gap
        raw_gap = row['open'] - data.iloc[i-1]['close']
        
        # Daily Volatility Range
        daily_range = row['high'] - row['low']
        
        # Volatility-Scaled Gap (avoid division by zero)
        if daily_range > 0:
            vol_scaled_gap = raw_gap / daily_range
        else:
            vol_scaled_gap = 0
            
        # Intraday Trend Strength
        intraday_trend = (row['close'] - row['open']) / daily_range if daily_range > 0 else 0
        
        # Volume-Weighted Momentum Component
        comp1 = vol_scaled_gap * intraday_trend * row['volume']
        
        # === Component 2: Liquidity-Efficiency Factor with Gap Interaction ===
        # Price Movement Efficiency
        price_efficiency = abs(row['close'] - data.iloc[i-1]['close']) / daily_range if daily_range > 0 else 0
        
        # Volume-to-Gap Liquidity (avoid division by zero)
        gap_magnitude = abs(raw_gap)
        if gap_magnitude > 0:
            volume_to_gap = row['volume'] / gap_magnitude
        else:
            volume_to_gap = 0
            
        comp2 = price_efficiency * volume_to_gap
        
        # === Component 3: Multi-Timeframe Acceleration with Volume-Weighted Gap Effects ===
        # Short-term Acceleration (2-day)
        short_term_accel = (row['close'] - data.iloc[i-2]['close']) - (data.iloc[i-2]['close'] - data.iloc[i-4]['close'])
        
        # Medium-term Acceleration (5-day)
        medium_term_accel = (row['close'] - data.iloc[i-5]['close']) - (data.iloc[i-5]['close'] - data.iloc[i-10]['close'])
        
        # Acceleration Divergence
        accel_divergence = short_term_accel - medium_term_accel
        
        # Volume-Weighted Gap Interaction
        comp3 = accel_divergence * raw_gap * row['volume']
        
        # === Component 4: Intraday Recovery Strength with Volume-Adjusted Liquidity ===
        # Raw Opening Gap (same as above)
        # raw_gap already calculated
        
        # Intraday Recovery Performance
        intraday_recovery = row['close'] - row['open']
        
        # Recovery-to-Gap Ratio (avoid division by zero)
        if abs(raw_gap) > 0:
            recovery_ratio = intraday_recovery / raw_gap
        else:
            recovery_ratio = 0
            
        # Volume-based Liquidity Change
        volume_liquidity = row['volume'] / data.iloc[i-1]['volume'] if data.iloc[i-1]['volume'] > 0 else 1
        
        comp4 = recovery_ratio * volume_liquidity
        
        # === Combine Components ===
        # Normalize and combine components with equal weights
        factor_value = 0.25 * comp1 + 0.25 * comp2 + 0.25 * comp3 + 0.25 * comp4
        factor_values[date] = factor_value
    
    return factor_values
