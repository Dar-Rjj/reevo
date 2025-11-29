import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor combining intraday patterns, volatility breakouts, 
    liquidity resilience, and opening session quality.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    for current_date in data.index:
        current_idx = data.index.get_loc(current_date)
        if current_idx < 20:  # Need sufficient history
            factor.loc[current_date] = 0
            continue
            
        # Get current day data
        current = data.iloc[current_idx]
        # Get historical data (past 20 days)
        hist_data = data.iloc[max(0, current_idx-20):current_idx]
        
        # 1. Intraday Price Pattern Efficiency
        morning_high = current['high'] - current['open']
        morning_low = current['open'] - current['low']
        morning_range = morning_high + morning_low
        
        # Afternoon reversal (from morning extremes to close)
        afternoon_reversal_high = current['high'] - current['close'] if current['close'] < current['high'] else 0
        afternoon_reversal_low = current['close'] - current['low'] if current['close'] > current['low'] else 0
        total_afternoon_reversal = afternoon_reversal_high + afternoon_reversal_low
        
        # Reversal efficiency ratio
        reversal_efficiency = total_afternoon_reversal / morning_range if morning_range > 0 else 0
        
        # Volume pattern validation (using amount as proxy for volume intensity)
        avg_morning_volume = hist_data['amount'].mean() * 0.4  # Assume 40% of volume in morning
        volume_consistency = min(current['amount'] / (avg_morning_volume * 2.5) if avg_morning_volume > 0 else 0, 2.0)
        
        intraday_factor = reversal_efficiency * volume_consistency
        
        # 2. Volatility Breakout Quality
        current_range = current['high'] - current['low']
        avg_range_5d = hist_data.iloc[-5:]['high'].subtract(hist_data.iloc[-5:]['low']).mean()
        
        # Price compression measurement
        range_ratio = current_range / avg_range_5d if avg_range_5d > 0 else 1.0
        
        # Count consecutive low-range days
        recent_ranges = [data.iloc[i]['high'] - data.iloc[i]['low'] for i in range(current_idx-5, current_idx)]
        avg_recent_range = np.mean(recent_ranges) if recent_ranges else current_range
        compression_days = sum(1 for r in recent_ranges if r < avg_recent_range * 0.7)
        
        # Breakout strength assessment
        range_mid = (current['high'] + current['low']) / 2
        close_position = (current['close'] - range_mid) / (current_range / 2) if current_range > 0 else 0
        
        breakout_strength = abs(close_position) * (1 + compression_days * 0.1)
        
        # Volume expansion confirmation
        avg_volume_5d = hist_data.iloc[-5:]['volume'].mean()
        volume_expansion = current['volume'] / avg_volume_5d if avg_volume_5d > 0 else 1.0
        
        volatility_factor = breakout_strength * min(volume_expansion, 3.0)
        
        # 3. Liquidity Shock Resilience
        daily_return = abs(current['close'] - current['open']) / current['open'] if current['open'] > 0 else 0
        
        # Price impact efficiency
        return_per_volume = daily_return / current['volume'] if current['volume'] > 0 else 0
        
        # Compare with normal periods
        hist_returns = [abs(data.iloc[i]['close'] - data.iloc[i]['open']) / data.iloc[i]['open'] 
                       for i in range(current_idx-10, current_idx) if data.iloc[i]['open'] > 0]
        hist_volumes = [data.iloc[i]['volume'] for i in range(current_idx-10, current_idx)]
        
        if len(hist_returns) > 5:
            avg_return_per_volume = np.mean([r/v for r, v in zip(hist_returns, hist_volumes) if v > 0])
            impact_efficiency = avg_return_per_volume / return_per_volume if return_per_volume > 0 else 1.0
        else:
            impact_efficiency = 1.0
        
        # Recovery pattern analysis (using intraday recovery)
        max_deviation = max(abs(current['high'] - current['open']), abs(current['open'] - current['low']))
        recovery_efficiency = 1 - (abs(current['close'] - current['open']) / max_deviation) if max_deviation > 0 else 1.0
        
        liquidity_factor = impact_efficiency * recovery_efficiency
        
        # 4. Opening Session Quality
        daily_range = current['high'] - current['low']
        open_position = (current['open'] - current['low']) / daily_range if daily_range > 0 else 0.5
        
        # Gap magnitude relative to recent volatility
        prev_close = data.iloc[current_idx-1]['close'] if current_idx > 0 else current['open']
        gap_magnitude = abs(current['open'] - prev_close) / prev_close if prev_close > 0 else 0
        
        hist_volatility = hist_data['close'].pct_change().std() * np.sqrt(252)
        gap_relative = gap_magnitude / (hist_volatility / 16) if hist_volatility > 0 else 1.0
        
        auction_efficiency = 1 - abs(open_position - 0.5) * min(gap_relative, 2.0)
        
        # Early momentum validation (using first hour proxy)
        first_hour_range = abs(current['high'] - current['open']) + abs(current['open'] - current['low'])
        early_momentum = (current['close'] - current['open']) / (first_hour_range / 2) if first_hour_range > 0 else 0
        
        opening_factor = auction_efficiency * (1 + early_momentum)
        
        # Combine all sub-factors with equal weighting
        combined_factor = (intraday_factor + volatility_factor + liquidity_factor + opening_factor) / 4
        
        factor.loc[current_date] = combined_factor
    
    # Normalize the factor
    if len(factor) > 20:
        rolling_mean = factor.rolling(window=20, min_periods=10).mean()
        rolling_std = factor.rolling(window=20, min_periods=10).std()
        factor = (factor - rolling_mean) / rolling_std
    
    return factor.fillna(0)
