import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Calculate rolling windows for various lookback periods
    for i in range(len(data)):
        if i < 10:  # Need enough data for calculations
            alpha.iloc[i] = 0
            continue
            
        current = data.iloc[i]
        prev = data.iloc[i-1] if i > 0 else current
        
        # Intraday Range-Liquidity Efficiency Factor
        # Range Compression Analysis
        true_range = max(current['high'] - current['low'], 
                        abs(current['high'] - prev['close']), 
                        abs(current['low'] - prev['close']))
        
        # Rolling high/low for past 5 days (t-4 to t)
        window_high = data['high'].iloc[max(0, i-4):i+1].max()
        window_low = data['low'].iloc[max(0, i-4):i+1].min()
        range_compression = true_range / (window_high - window_low) if (window_high - window_low) > 0 else 1
        
        # Intraday Momentum Divergence
        gap_momentum = (current['open'] - prev['close']) / (current['high'] - current['low']) if (current['high'] - current['low']) > 0 else 0
        position_efficiency = ((current['close'] - current['low']) - (current['open'] - current['low'])) / (current['high'] - current['low']) if (current['high'] - current['low']) > 0 else 0
        range_momentum = (current['high'] - current['low']) / (prev['high'] - prev['low']) if (prev['high'] - prev['low']) > 0 else 1
        
        # Liquidity Efficiency Signals
        volume_window = data['volume'].iloc[max(0, i-10):i].median()
        volume_intensity = current['volume'] / volume_window if volume_window > 0 else 1
        
        amount_efficiency = abs(current['close'] - prev['close']) / current['amount'] if current['amount'] > 0 else 0
        
        liquidity_accumulation = (current['amount'] - prev['amount']) / prev['amount'] if prev['amount'] > 0 else 0
        
        # Price Rejection Analysis
        upper_rejection = (current['high'] - current['close']) * current['volume']
        lower_rejection = (current['close'] - current['low']) * current['volume']
        net_rejection = upper_rejection - lower_rejection
        
        # Alpha Synthesis for Intraday Range-Liquidity Efficiency
        core_efficiency = gap_momentum * position_efficiency * amount_efficiency
        range_momentum_enhancement = core_efficiency * range_momentum
        liquidity_alignment = range_momentum_enhancement * volume_intensity * liquidity_accumulation
        rejection_filter = liquidity_alignment * net_rejection
        intraday_alpha = rejection_filter * range_compression
        
        # Overnight Gap-Range Persistence Factor
        # Gap Analysis
        overnight_gap = (current['open'] - prev['close']) / prev['close'] if prev['close'] > 0 else 0
        gap_persistence = (1 if current['low'] > prev['close'] else 0) - (1 if current['high'] < prev['close'] else 0)
        gap_efficiency = abs(current['open'] - prev['close']) / current['amount'] if current['amount'] > 0 else 0
        
        # Intraday Range Dynamics
        range_position = (current['close'] - current['low']) / (current['high'] - current['low']) if (current['high'] - current['low']) > 0 else 0.5
        range_efficiency = abs(current['close'] - prev['close']) / (current['high'] - current['low']) if (current['high'] - current['low']) > 0 else 0
        
        # Volume-Liquidity Alignment
        volume_surge = volume_intensity  # Reuse from above
        liquidity_momentum = liquidity_accumulation  # Reuse from above
        price_rejection = net_rejection  # Reuse from above
        
        # Alpha Synthesis for Overnight Gap-Range Persistence
        core_persistence = overnight_gap * gap_persistence * gap_efficiency
        range_enhancement = core_persistence * range_position * range_efficiency
        liquidity_confirmation = range_enhancement * volume_surge * liquidity_momentum
        overnight_alpha = liquidity_confirmation * price_rejection * range_compression
        
        # Volatility-Regime Range-Liquidity Divergence
        # Volatility Analysis
        volatility_regime = true_range / data['high'].iloc[max(0, i-5):i].subtract(data['low'].iloc[max(0, i-5):i]).median() if i >= 5 else 1
        
        # Intraday Efficiency
        position_divergence = position_efficiency  # Reuse from above
        movement_efficiency = amount_efficiency  # Reuse from above
        range_efficiency_div = range_efficiency  # Reuse from above
        
        # Liquidity Dynamics
        volume_intensity_div = volume_intensity  # Reuse from above
        amount_momentum = liquidity_accumulation  # Reuse from above
        liquidity_efficiency = amount_efficiency  # Reuse from above
        
        # Alpha Synthesis for Volatility-Regime
        core_divergence = position_divergence * range_momentum
        efficiency_enhancement = core_divergence * movement_efficiency * range_efficiency_div
        liquidity_confirmation_div = efficiency_enhancement * volume_intensity_div * amount_momentum
        volatility_alpha = liquidity_confirmation_div * volatility_regime * liquidity_efficiency
        
        # Combine all three alpha factors with equal weighting
        combined_alpha = (intraday_alpha + overnight_alpha + volatility_alpha) / 3
        
        alpha.iloc[i] = combined_alpha
    
    # Handle any remaining NaN values
    alpha = alpha.fillna(0)
    
    return alpha
