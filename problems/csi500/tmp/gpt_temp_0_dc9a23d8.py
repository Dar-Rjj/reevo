import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Microstructure Momentum Decay factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    # Rolling window parameters
    short_window = 5
    medium_window = 20
    long_window = 60
    
    for current_date in df.index:
        current_idx = df.index.get_loc(current_date)
        
        # Only calculate if we have enough historical data
        if current_idx < long_window:
            factor_values.loc[current_date] = 0
            continue
            
        # Get historical data up to current date
        hist_data = df.iloc[:current_idx+1]
        
        # 1. Price Impact Asymmetry
        # Bid-ask spread estimation via high-low range
        spread_ratio = (hist_data['high'] - hist_data['low']) / hist_data['close']
        avg_spread = spread_ratio.rolling(window=short_window).mean().iloc[-1]
        
        # Large trade clustering detection (volume spikes)
        volume_zscore = (hist_data['volume'] - hist_data['volume'].rolling(window=medium_window).mean()) / hist_data['volume'].rolling(window=medium_window).std()
        volume_spike = volume_zscore.rolling(window=3).max().iloc[-1]
        
        # Directional persistence after clustered trades
        recent_returns = hist_data['close'].pct_change().tail(short_window)
        persistence = recent_returns.autocorr(lag=1) if len(recent_returns) > 1 else 0
        
        price_impact = avg_spread * volume_spike * (1 + persistence)
        
        # 2. Momentum Decay Timing
        # Short-term momentum reversal threshold identification
        momentum_5d = hist_data['close'].pct_change(periods=5).iloc[-1]
        momentum_1d = hist_data['close'].pct_change().iloc[-1]
        reversal_signal = -np.sign(momentum_5d) * abs(momentum_1d)
        
        # Volume acceleration preceding momentum exhaustion
        volume_accel = (hist_data['volume'].pct_change().rolling(window=3).mean().iloc[-1] - 
                       hist_data['volume'].pct_change().rolling(window=10).mean().iloc[-1])
        
        # Intraday time-of-day momentum decay patterns (using high-low range as proxy)
        tod_efficiency = (hist_data['close'].iloc[-1] - hist_data['open'].iloc[-1]) / (hist_data['high'].iloc[-1] - hist_data['low'].iloc[-1])
        
        momentum_decay = reversal_signal * volume_accel * tod_efficiency
        
        # 3. Liquidity Regime Microstructure
        # Quote intensity via volume/price change correlation
        recent_corr = hist_data['volume'].tail(short_window).corr(hist_data['close'].pct_change().tail(short_window))
        quote_intensity = recent_corr if not np.isnan(recent_corr) else 0
        
        # Large order absorption capacity (volume persistence)
        volume_persistence = hist_data['volume'].autocorr(lag=1)
        
        # Hidden liquidity detection via price continuity
        price_gaps = (hist_data['open'] - hist_data['close'].shift(1)).abs() / hist_data['close'].shift(1)
        price_continuity = 1 - price_gaps.rolling(window=short_window).mean().iloc[-1]
        
        liquidity_regime = quote_intensity * volume_persistence * price_continuity
        
        # 4. Price Discovery Efficiency
        # New information incorporation speed (variance ratio decay)
        daily_returns = hist_data['close'].pct_change().dropna()
        if len(daily_returns) >= 10:
            var_1d = daily_returns.var()
            var_5d = daily_returns.rolling(window=5).sum().var()
            variance_ratio = var_5d / (5 * var_1d) if var_1d > 0 else 1
        else:
            variance_ratio = 1
        info_speed = 1 - min(variance_ratio, 2) / 2
        
        # Overnight gap vs intraday range efficiency
        overnight_gap = (hist_data['open'] - hist_data['close'].shift(1)).abs() / hist_data['close'].shift(1)
        intraday_range = (hist_data['high'] - hist_data['low']) / hist_data['close']
        gap_efficiency = 1 - (overnight_gap.rolling(window=short_window).mean().iloc[-1] / 
                             intraday_range.rolling(window=short_window).mean().iloc[-1])
        
        # Micro-trend fragmentation (high-frequency mean reversion)
        intraday_changes = (hist_data['close'] - hist_data['open']) / (hist_data['high'] - hist_data['low'])
        fragmentation = intraday_changes.rolling(window=short_window).std().iloc[-1]
        
        price_discovery = info_speed * gap_efficiency * (1 - fragmentation)
        
        # 5. Adaptive Signal Weighting
        # Weight by recent microstructure noise level
        noise_level = hist_data['close'].pct_change().rolling(window=short_window).std().iloc[-1]
        
        # Scale by current liquidity regime intensity
        current_volume = hist_data['volume'].iloc[-1] / hist_data['volume'].rolling(window=medium_window).mean().iloc[-1]
        
        # Adjust for intraday momentum decay phase
        momentum_strength = abs(momentum_5d)
        
        # Combine all components with adaptive weights
        base_signal = (price_impact * 0.2 + 
                      momentum_decay * 0.3 + 
                      liquidity_regime * 0.25 + 
                      price_discovery * 0.25)
        
        # Apply adaptive weighting
        adaptive_weight = (1 / (1 + noise_level)) * current_volume * (1 - momentum_strength)
        final_factor = base_signal * adaptive_weight
        
        factor_values.loc[current_date] = final_factor
    
    # Normalize the factor
    if len(factor_values.dropna()) > 0:
        factor_values = (factor_values - factor_values.mean()) / factor_values.std()
    
    return factor_values.fillna(0)
