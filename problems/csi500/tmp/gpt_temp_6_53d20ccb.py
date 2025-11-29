import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Volatility-Adjusted Price Momentum
    def intraday_vol_adj_momentum(df, N=10, M=5):
        # Price momentum
        momentum = df['close'] - df['close'].shift(N)
        
        # Intraday volatility (daily range)
        daily_range = df['high'] - df['low']
        
        # Rolling volatility
        rolling_vol = daily_range.rolling(M).std()
        
        # Avoid division by zero
        rolling_vol = rolling_vol.replace(0, np.nan)
        
        return momentum / rolling_vol

    # High-Low Breakout Efficiency
    def hl_breakout_efficiency(df, K=5):
        # Identify breakout days
        prev_high = df['high'].shift(1).rolling(K).max()
        prev_low = df['low'].shift(1).rolling(K).min()
        
        high_breakout = (df['high'] > prev_high).astype(int)
        low_breakout = (df['low'] < prev_low).astype(int)
        breakout_days = high_breakout + low_breakout
        
        # Absolute price movement (high-low range)
        abs_movement = (df['high'] - df['low']).rolling(K).sum()
        
        # Total price path (close-to-close changes)
        close_changes = df['close'].diff().abs()
        total_path = close_changes.rolling(K).sum()
        
        # Avoid division by zero
        total_path = total_path.replace(0, np.nan)
        
        efficiency = abs_movement / total_path
        return breakout_days * efficiency

    # Volume-Price Divergence Factor
    def volume_price_divergence(df, window=10):
        def rolling_slope(series, window):
            x = np.arange(window)
            slopes = []
            for i in range(len(series)):
                if i < window - 1:
                    slopes.append(np.nan)
                else:
                    y = series.iloc[i-window+1:i+1].values
                    if len(y) == window:
                        slope = np.polyfit(x, y, 1)[0]
                        slopes.append(slope)
                    else:
                        slopes.append(np.nan)
            return pd.Series(slopes, index=series.index)
        
        price_slope = rolling_slope(df['close'], window)
        volume_slope = rolling_slope(df['volume'], window)
        
        return price_slope * volume_slope

    # Acceleration-Deceleration Oscillator
    def acceleration_deceleration(df, smooth_window=5):
        # Price velocity (first derivative)
        price_velocity = df['close'].diff()
        
        # Smooth velocity
        smooth_velocity = price_velocity.rolling(smooth_window).mean()
        
        # Acceleration (second derivative)
        acceleration = smooth_velocity.diff()
        
        return smooth_velocity * acceleration

    # Relative Strength Pressure
    def relative_strength_pressure(df, vol_window=20):
        # Stock's daily return
        stock_return = df['close'].pct_change()
        
        # Market return proxy (cross-sectional average)
        market_return = stock_return.groupby(level=0).transform('mean')
        
        # Relative performance
        relative_perf = stock_return - market_return
        
        # Volume surge
        avg_volume = df['volume'].rolling(vol_window).mean()
        volume_surge = df['volume'] / avg_volume
        
        return relative_perf * volume_surge

    # Price-Volume Correlation Reversal
    def price_volume_corr_reversal(df, corr_window=20, std_window=10):
        # Price and volume changes
        price_changes = df['close'].pct_change()
        volume_changes = df['volume'].pct_change()
        
        # Rolling correlation
        correlation = price_changes.rolling(corr_window).corr(volume_changes)
        
        # Find local maxima
        rolling_max = correlation.rolling(std_window, center=True).max()
        is_peak = (correlation == rolling_max) & (correlation > 0)
        
        # Rolling standard deviation for scaling
        corr_std = correlation.rolling(std_window).std()
        
        # Generate negative signal at peaks
        signal = -is_peak.astype(int) * correlation.abs()
        
        return signal

    # Intraday Momentum Persistence
    def intraday_momentum_persistence(df, lookback=5):
        # Assuming midday is approximated by (high + low) / 2
        midday_price = (df['high'] + df['low']) / 2
        
        # Morning session strength (open to midday)
        morning_strength = (midday_price - df['open']) / df['open']
        
        # Historical morning patterns
        hist_morning = morning_strength.rolling(lookback).mean()
        
        # Afternoon continuation (midday to close)
        afternoon_continuation = (df['close'] - midday_price) / midday_price
        
        return morning_strength / hist_morning * afternoon_continuation

    # Liquidity-Adjusted Trend Following
    def liquidity_adj_trend(df, short_window=5, long_window=20, vol_window=10):
        # Multiple moving averages
        short_ma = df['close'].rolling(short_window).mean()
        long_ma = df['close'].rolling(long_window).mean()
        
        # Trend direction and strength
        trend_direction = np.sign(short_ma - long_ma)
        trend_strength = (short_ma - long_ma).abs() / long_ma
        
        # Dollar volume
        dollar_volume = df['close'] * df['volume']
        
        # Volume-based liquidity (rolling percentile)
        liquidity_score = dollar_volume.rolling(vol_window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == vol_window else np.nan
        )
        
        return trend_direction * trend_strength * liquidity_score

    # Combine all factors with equal weights
    factors = [
        intraday_vol_adj_momentum(df),
        hl_breakout_efficiency(df),
        volume_price_divergence(df),
        acceleration_deceleration(df),
        relative_strength_pressure(df),
        price_volume_corr_reversal(df),
        intraday_momentum_persistence(df),
        liquidity_adj_trend(df)
    ]
    
    # Normalize and combine
    normalized_factors = []
    for factor in factors:
        if factor.notna().any():
            # Z-score normalization
            normalized = (factor - factor.mean()) / factor.std()
            normalized_factors.append(normalized)
    
    # Equal-weighted combination
    combined_factor = pd.concat(normalized_factors, axis=1).mean(axis=1)
    
    return combined_factor
