import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 20:  # Need sufficient history for calculations
            factor_values.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]  # Only use current and past data
        
        # 1. Intraday Momentum Reversal Divergence
        try:
            # Estimate first hour high/low using opening patterns
            first_hour_high = current_data['high'].rolling(window=5).mean().iloc[-1]
            first_hour_low = current_data['low'].rolling(window=5).mean().iloc[-1]
            early_reversal = (current_data['open'].iloc[-1] - first_hour_low) / (first_hour_high - current_data['open'].iloc[-1] + 1e-8)
            
            # Estimate last hour high/low using closing patterns
            last_hour_high = current_data['high'].rolling(window=3).mean().iloc[-1]
            last_hour_low = current_data['low'].rolling(window=3).mean().iloc[-1]
            late_reversal = (current_data['close'].iloc[-1] - last_hour_low) / (last_hour_high - current_data['close'].iloc[-1] + 1e-8)
            
            # Volume concentration (simplified)
            recent_volume = current_data['volume'].iloc[-5:].mean()
            older_volume = current_data['volume'].iloc[-20:-5].mean()
            volume_concentration = recent_volume / (older_volume + 1e-8)
            
            reversal_strength = (early_reversal + late_reversal) * volume_concentration
        except:
            reversal_strength = 0
        
        # 2. Opening Auction Imbalance Persistence
        try:
            # Pre-open pressure estimation
            prev_high = current_data['high'].iloc[-2]
            prev_low = current_data['low'].iloc[-2]
            pre_open_pressure = (current_data['open'].iloc[-1] - prev_low) / (prev_high - current_data['open'].iloc[-1] + 1e-8)
            
            # Consecutive imbalance days
            recent_opens = current_data['open'].iloc[-5:]
            recent_highs = current_data['high'].iloc[-5:]
            recent_lows = current_data['low'].iloc[-5:]
            imbalance_days = sum((recent_opens - recent_lows) / (recent_highs - recent_opens + 1e-8) > 0.5)
            
            # Volume intensity
            opening_volume = current_data['volume'].iloc[-1]
            prev_opening_volume = current_data['volume'].iloc[-2]
            volume_intensity = opening_volume / (prev_opening_volume + 1e-8)
            
            imbalance_signal = imbalance_days * volume_intensity
        except:
            imbalance_signal = 0
        
        # 3. Volatility Compression Breakout
        try:
            # Range compression
            current_range = current_data['high'].iloc[-1] - current_data['low'].iloc[-1]
            avg_range = (current_data['high'] - current_data['low']).rolling(window=10).mean().iloc[-1]
            range_compression = current_range / (avg_range + 1e-8)
            
            # Compression duration (days with range below average)
            recent_ranges = current_data['high'].iloc[-10:] - current_data['low'].iloc[-10:]
            compression_duration = sum(recent_ranges < avg_range)
            
            # Breakout volume ratio
            breakout_volume = current_data['volume'].iloc[-1]
            compression_volume = current_data['volume'].iloc[-10:].mean()
            volume_ratio = breakout_volume / (compression_volume + 1e-8)
            
            breakout_strength = compression_duration * volume_ratio
        except:
            breakout_strength = 0
        
        # 4. Price-Volume Fractal Dimension
        try:
            # Price fractal (simplified)
            price_range = current_data['high'].iloc[-5:].max() - current_data['low'].iloc[-5:].min()
            price_fractal = np.log(price_range + 1e-8) / np.log(5)
            
            # Volume fractal
            volume_variance = current_data['volume'].iloc[-5:].var()
            volume_fractal = np.log(volume_variance + 1e-8) / np.log(5)
            
            dimension_difference = price_fractal - volume_fractal
            
            # Fractal persistence (correlation of recent patterns)
            recent_price_fractals = []
            recent_volume_fractals = []
            for j in range(3):
                if i - j*5 >= 5:
                    sub_range = current_data.iloc[i-j*5-4:i-j*5+1]
                    p_range = sub_range['high'].max() - sub_range['low'].min()
                    v_var = sub_range['volume'].var()
                    recent_price_fractals.append(np.log(p_range + 1e-8) / np.log(5))
                    recent_volume_fractals.append(np.log(v_var + 1e-8) / np.log(5))
            
            if len(recent_price_fractals) > 1:
                fractal_persistence = np.corrcoef(recent_price_fractals, recent_volume_fractals)[0,1]
            else:
                fractal_persistence = 0
                
            pattern_signal = dimension_difference * fractal_persistence
        except:
            pattern_signal = 0
        
        # Combine all components with weights
        factor_value = (
            0.25 * reversal_strength +
            0.20 * imbalance_signal +
            0.20 * breakout_strength +
            0.15 * pattern_signal
        )
        
        factor_values.iloc[i] = factor_value
    
    # Normalize the factor values
    factor_values = (factor_values - factor_values.mean()) / (factor_values.std() + 1e-8)
    
    return factor_values
