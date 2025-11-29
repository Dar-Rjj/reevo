import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Calculate Price Rejection Component
    # Compute Intraday Midpoint
    df['midpoint'] = (df['high'] + df['low']) / 2
    
    # Calculate Normalized Rejection Distance
    df['rejection_distance'] = df['close'] - df['midpoint']
    daily_range = df['high'] - df['low']
    df['normalized_rejection'] = df['rejection_distance'] / np.where(daily_range == 0, 1, daily_range)
    
    # Apply Momentum Direction Filter
    df['intraday_direction'] = np.sign(df['close'] - df['open'])
    df['rejection_signal'] = df['normalized_rejection'] * df['intraday_direction']
    
    # Calculate Volume-Volatility Divergence
    # Compute Volume Strength
    df['avg_volume_20d'] = df['volume'].rolling(window=20, min_periods=10).mean()
    df['volume_ratio'] = df['volume'] / np.where(df['avg_volume_20d'] == 0, 1, df['avg_volume_20d'])
    df['volume_momentum'] = df['volume'].pct_change(periods=5)
    
    # Calculate Volatility Adjustment
    df['range_volatility'] = (df['high'] - df['low']).rolling(window=5, min_periods=3).mean()
    df['return_volatility'] = df['close'].pct_change().rolling(window=10, min_periods=5).std()
    df['avg_volatility'] = (df['range_volatility'] + df['return_volatility']) / 2
    
    # Generate Volume Divergence Signal
    df['volume_strength'] = df['volume_ratio'] * df['volume_momentum']
    df['volume_volatility_composite'] = df['volume_strength'] * df['avg_volatility']
    df['volume_divergence'] = df['volume_volatility_composite'] - df['volume_volatility_composite'].rolling(window=5, min_periods=3).mean()
    
    # Detect Rejection-Volume Divergence Pattern
    # Compare Rejection and Volume Signals
    rolling_corr = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        if i >= 5:
            window_rejection = df['rejection_signal'].iloc[i-5:i]
            window_volume = df['volume_divergence'].iloc[i-5:i]
            if len(window_rejection) >= 3 and len(window_volume) >= 3:
                corr = window_rejection.corr(window_volume)
                rolling_corr.iloc[i] = corr if not np.isnan(corr) else 0
            else:
                rolling_corr.iloc[i] = 0
    
    df['correlation_5d'] = rolling_corr
    
    # Apply Confirmation Logic
    strong_rejection = df['rejection_signal'].abs() > df['rejection_signal'].abs().rolling(window=10, min_periods=5).quantile(0.7)
    weak_volume_div = df['volume_divergence'].abs() < df['volume_divergence'].abs().rolling(window=10, min_periods=5).quantile(0.3)
    weak_rejection = df['rejection_signal'].abs() < df['rejection_signal'].abs().rolling(window=10, min_periods=5).quantile(0.3)
    strong_volume_div = df['volume_divergence'].abs() > df['volume_divergence'].abs().rolling(window=10, min_periods=5).quantile(0.7)
    
    df['divergence_pattern'] = 0
    df.loc[strong_rejection & weak_volume_div, 'divergence_pattern'] = df['rejection_signal']
    df.loc[weak_rejection & strong_volume_div, 'divergence_pattern'] = -df['volume_divergence']
    
    # Incorporate Efficiency-Based Filtering
    # Calculate Directional Price Efficiency
    df['bullish_efficiency'] = (df['high'] - df['close']) / np.where((df['high'] - df['low']) == 0, 1, (df['high'] - df['low']))
    df['bearish_efficiency'] = (df['close'] - df['low']) / np.where((df['high'] - df['low']) == 0, 1, (df['high'] - df['low']))
    df['net_efficiency'] = df['bullish_efficiency'] - df['bearish_efficiency']
    
    # Apply Volume Surge Filter
    df['volume_median_10d'] = df['volume'].rolling(window=10, min_periods=5).median()
    df['volume_surge'] = df['volume'] > (1.5 * df['volume_median_10d'])
    
    # Combine Components with Multiplicative Logic
    # Multiply rejection signal by volume divergence
    df['base_signal'] = df['rejection_signal'] * df['volume_divergence']
    
    # Apply efficiency filter as conditional multiplier
    df['efficiency_multiplier'] = 1.0
    df.loc[df['volume_surge'], 'efficiency_multiplier'] = 1 + df['net_efficiency']
    
    # Incorporate volatility adjustment as scaling factor
    vol_scaling = 1 / (1 + df['avg_volatility'])
    df['combined_signal'] = df['base_signal'] * df['efficiency_multiplier'] * vol_scaling
    
    # Generate Final Alpha Signal
    # Apply Trend Confirmation
    df['price_momentum_5d'] = df['close'].pct_change(periods=5)
    df['trend_adjusted_signal'] = df['combined_signal'] * np.sign(df['price_momentum_5d'])
    
    # Smooth Signal
    df['smoothed_signal'] = df['trend_adjusted_signal'].ewm(span=3, min_periods=2).mean()
    
    # Apply Directional Consistency Check
    df['price_trend_2d'] = np.sign(df['close'].pct_change(periods=2))
    df['signal_direction'] = np.sign(df['smoothed_signal'])
    
    # Reverse signal if contradictory
    contradictory = (df['signal_direction'] != df['price_trend_2d']) & (df['price_trend_2d'] != 0)
    df['final_factor'] = df['smoothed_signal']
    df.loc[contradictory, 'final_factor'] = -df['smoothed_signal']
    
    return df['final_factor']
