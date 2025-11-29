import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Intraday Trend Persistence Factor
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['intraday_trend'] = (df['close'] - df['open']) / df['true_range']
    
    # Calculate rolling correlation of intraday trends
    trend_corr = df['intraday_trend'].rolling(window=5, min_periods=3).corr(df['intraday_trend'].shift(1))
    hist_vol = df['returns'].rolling(window=20, min_periods=10).std()
    factor1 = trend_corr * df['intraday_trend'] / hist_vol
    
    # Volume-Weighted Price Acceleration
    df['returns_3d_diff'] = df['returns'].diff(3)
    avg_volume = df['volume'].rolling(window=10, min_periods=5).mean()
    factor2 = (df['returns_3d_diff'] * df['volume']) / avg_volume
    
    # Relative Gap Momentum Factor
    df['price_gap'] = abs((df['open'] - df['close'].shift(1)) / df['close'].shift(1))
    gap_mean = df['price_gap'].rolling(window=20, min_periods=10).mean()
    gap_std = df['price_gap'].rolling(window=20, min_periods=10).std()
    factor3 = (df['price_gap'] - gap_mean) / gap_std
    
    # Volatility-Regressed Return Reversal
    vol_10d = df['returns'].rolling(window=10, min_periods=5).std()
    
    def rolling_residual(returns, volatility, window=15):
        residuals = pd.Series(index=returns.index, dtype=float)
        for i in range(window, len(returns)):
            if i >= window:
                window_returns = returns.iloc[i-window:i]
                window_vol = volatility.iloc[i-window:i]
                if len(window_returns) >= 5 and window_vol.notna().all():
                    try:
                        beta = np.cov(window_returns, window_vol)[0,1] / np.var(window_vol)
                        alpha = np.mean(window_returns) - beta * np.mean(window_vol)
                        residuals.iloc[i] = returns.iloc[i] - (alpha + beta * volatility.iloc[i])
                    except:
                        residuals.iloc[i] = 0
        return residuals
    
    factor4 = rolling_residual(df['returns'], vol_10d, window=15)
    
    # Liquidity-Adjusted Momentum Divergence
    ret_5d = df['close'].pct_change(5)
    ret_10d = df['close'].pct_change(10)
    price_momentum = ret_5d - ret_10d
    
    vol_5d_change = df['volume'].pct_change(5)
    factor5 = price_momentum * vol_5d_change
    
    # High-Low Compression Breakout Signal
    df['high_low_range'] = df['high'] - df['low']
    avg_range = df['high_low_range'].rolling(window=5, min_periods=3).mean()
    range_expansion = (df['high_low_range'] > 1.5 * avg_range).astype(int)
    price_direction = np.where(df['close'] > df['open'], 1, -1)
    factor6 = range_expansion * price_direction
    
    # Volume-Price Correlation Regime Change
    vol_price_corr = df['returns'].rolling(window=10, min_periods=5).corr(df['volume'])
    corr_change = vol_price_corr.diff()
    vol_ratio = df['volume'] / df['volume'].rolling(window=20, min_periods=10).mean()
    factor7 = corr_change * vol_ratio
    
    # Efficiency Ratio Weighted Momentum
    price_changes = df['close'].diff().abs()
    total_movement = price_changes.rolling(window=10, min_periods=5).sum()
    net_movement = abs(df['close'] - df['close'].shift(10))
    efficiency_ratio = net_movement / total_movement
    factor8 = df['close'].pct_change(5) * efficiency_ratio
    
    # Open-Close Relative Strength Indicator
    intraday_return = (df['close'] - df['open']) / df['open']
    daily_range = df['high'] - df['low']
    intraday_strength = intraday_return / (daily_range / df['open'])
    avg_strength = intraday_strength.rolling(window=10, min_periods=5).mean()
    factor9 = intraday_strength - avg_strength
    
    # Volume-Weighted Volatility Clustering
    daily_vol = df['returns'].abs()
    vol_autocorr = daily_vol.rolling(window=5, min_periods=3).corr(daily_vol.shift(1))
    volume_rank = df['volume'].rolling(window=20, min_periods=10).rank(pct=True)
    factor10 = vol_autocorr * df['volume'] * volume_rank
    
    # Combine factors (equal weighting)
    factors = pd.DataFrame({
        'f1': factor1, 'f2': factor2, 'f3': factor3, 'f4': factor4,
        'f5': factor5, 'f6': factor6, 'f7': factor7, 'f8': factor8,
        'f9': factor9, 'f10': factor10
    })
    
    # Remove any potential future data and handle NaN values
    for col in factors.columns:
        factors[col] = factors[col].shift(1)  # Ensure no lookahead
    
    combined_factor = factors.mean(axis=1, skipna=True)
    
    return combined_factor
