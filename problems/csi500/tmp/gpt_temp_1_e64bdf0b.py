import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor mining function that generates novel factors
    using price, volume, and volatility patterns.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    for date in data.index:
        current_idx = data.index.get_loc(date)
        
        # Only use current and past data
        current_data = data.iloc[:current_idx+1]
        
        if len(current_data) < 20:  # Minimum data requirement
            factor_values.loc[date] = 0
            continue
            
        try:
            # Price-Based Momentum Factors
            # Acceleration-Adjusted Momentum
            mom_5 = current_data['close'].iloc[-1] / current_data['close'].iloc[-6] - 1
            mom_10 = current_data['close'].iloc[-1] / current_data['close'].iloc[-11] - 1
            mom_20 = current_data['close'].iloc[-1] / current_data['close'].iloc[-21] - 1
            
            # Acceleration as momentum change
            accel_5_10 = mom_5 - mom_10
            accel_10_20 = mom_10 - mom_20
            
            # Combined momentum and acceleration signal
            momentum_factor = (mom_5 + 0.7 * mom_10 + 0.3 * mom_20 + 
                             0.5 * accel_5_10 + 0.3 * accel_10_20)
            
            # Multi-Timeframe Momentum Divergence
            short_mom = current_data['close'].iloc[-1] / current_data['close'].iloc[-6] - 1
            medium_mom = current_data['close'].iloc[-1] / current_data['close'].iloc[-16] - 1
            momentum_divergence = short_mom - medium_mom
            
            # Volume-Price Interaction Factors
            # Volume-Weighted Price Efficiency
            recent_data = current_data.tail(10)
            price_efficiency = []
            for i in range(len(recent_data)-1):
                price_move = abs(recent_data['close'].iloc[i+1] / recent_data['close'].iloc[i] - 1)
                volume_ratio = recent_data['volume'].iloc[i] / recent_data['volume'].iloc[i:i+5].mean()
                if volume_ratio > 0:
                    efficiency = price_move / volume_ratio
                    price_efficiency.append(efficiency)
            
            vw_efficiency = np.mean(price_efficiency) if price_efficiency else 0
            
            # Volume-Cluster Breakout Signals
            volume_ma_5 = current_data['volume'].tail(5).mean()
            volume_ma_20 = current_data['volume'].tail(20).mean()
            volume_cluster_ratio = volume_ma_5 / volume_ma_20 if volume_ma_20 > 0 else 1
            
            recent_range = (current_data['high'].tail(5).max() - current_data['low'].tail(5).min()) / current_data['close'].tail(5).mean()
            breakout_strength = volume_cluster_ratio * recent_range
            
            # Intraday Pattern Factors
            # Opening Gap Reversal Strength
            if len(current_data) >= 2:
                prev_close = current_data['close'].iloc[-2]
                current_open = current_data['open'].iloc[-1]
                gap_magnitude = (current_open / prev_close - 1) if prev_close > 0 else 0
                
                # Intraday recovery (how much of gap was recovered)
                current_high = current_data['high'].iloc[-1]
                current_low = current_data['low'].iloc[-1]
                current_close = current_data['close'].iloc[-1]
                
                if gap_magnitude > 0:  # Gap up
                    recovery = (current_high - current_open) / (current_high - current_low) if (current_high - current_low) > 0 else 0
                else:  # Gap down
                    recovery = (current_open - current_low) / (current_high - current_low) if (current_high - current_low) > 0 else 0
                
                gap_reversal = gap_magnitude * recovery
            else:
                gap_reversal = 0
            
            # Close Position Persistence
            recent_closes = []
            for i in range(min(5, len(current_data))):
                daily_range = current_data['high'].iloc[-(i+1)] - current_data['low'].iloc[-(i+1)]
                if daily_range > 0:
                    close_pos = (current_data['close'].iloc[-(i+1)] - current_data['low'].iloc[-(i+1)]) / daily_range
                    recent_closes.append(close_pos)
            
            close_persistence = 1 - np.std(recent_closes) if recent_closes else 0.5
            
            # Volatility-Based Factors
            # High-Low Range Persistence
            ranges = []
            for i in range(min(10, len(current_data))):
                daily_range = (current_data['high'].iloc[-(i+1)] - current_data['low'].iloc[-(i+1)]) / current_data['close'].iloc[-(i+1)]
                ranges.append(daily_range)
            
            if len(ranges) >= 5:
                range_corr = np.corrcoef(range(len(ranges[:5])), ranges[:5])[0,1] if not np.isnan(np.corrcoef(range(len(ranges[:5])), ranges[:5])[0,1]) else 0
                range_persistence = abs(range_corr)
            else:
                range_persistence = 0
            
            # Intraday Volatility Compression
            recent_ranges = ranges[:5] if len(ranges) >= 5 else ranges
            range_std = np.std(recent_ranges) if recent_ranges else 0
            range_mean = np.mean(recent_ranges) if recent_ranges else 0
            volatility_compression = range_mean / (range_std + 1e-6) if range_std > 0 else 1
            
            # Combine all factors with weights
            final_factor = (
                0.25 * momentum_factor +
                0.15 * momentum_divergence +
                0.20 * vw_efficiency +
                0.10 * breakout_strength +
                0.10 * gap_reversal +
                0.10 * close_persistence +
                0.05 * range_persistence +
                0.05 * volatility_compression
            )
            
            factor_values.loc[date] = final_factor
            
        except (IndexError, KeyError, ZeroDivisionError):
            factor_values.loc[date] = 0
    
    return factor_values
