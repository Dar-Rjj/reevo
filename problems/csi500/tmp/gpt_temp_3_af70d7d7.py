import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Dynamics & Multi-Dimensional Momentum Coupling Alpha Factor
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Fractal Analysis
    # Calculate Fractal Dimensions using High-Low-Close triplets
    def fractal_dimension(high, low, close, window):
        """Calculate fractal dimension using high-low-close price movements"""
        # Price range normalized by average price
        price_range = (high - low) / ((high + low + close) / 3)
        
        # Calculate the sum of absolute price changes
        price_changes = close.diff().abs()
        
        # Calculate the rescaled range (Hurst-like approach)
        cumulative_deviation = (close - close.rolling(window=window, min_periods=1).mean()).cumsum()
        range_series = cumulative_deviation.rolling(window=window).max() - cumulative_deviation.rolling(window=window).min()
        std_series = close.rolling(window=window).std()
        
        # Avoid division by zero
        std_series = std_series.replace(0, np.nan)
        hurst_exp = np.log(range_series / std_series) / np.log(window)
        fractal_dim = 2 - hurst_exp
        
        return fractal_dim.fillna(method='ffill')
    
    # Compute fractal dimensions for different timeframes
    data['fractal_3d'] = fractal_dimension(data['high'], data['low'], data['close'], 3)
    data['fractal_8d'] = fractal_dimension(data['high'], data['low'], data['close'], 8)
    
    # Fractal Momentum Divergence
    data['fractal_momentum'] = data['fractal_3d'] - data['fractal_8d']
    data['fractal_acceleration'] = data['fractal_momentum'].diff(3)
    
    # Volume Fractal Patterns
    # Volume clustering analysis
    def volume_clustering(volume, window=5):
        """Analyze volume clustering patterns"""
        # Volume z-score to identify clusters
        volume_mean = volume.rolling(window=window).mean()
        volume_std = volume.rolling(window=window).std()
        volume_std = volume_std.replace(0, np.nan)
        volume_z = (volume - volume_mean) / volume_std
        
        # Cluster persistence (consecutive high/low volume days)
        high_volume_cluster = (volume_z > 1).astype(int)
        cluster_persistence = high_volume_cluster.rolling(window=3).sum()
        
        # Cluster intensity (magnitude of volume deviations)
        cluster_intensity = volume_z.rolling(window=window).std()
        
        return cluster_persistence.fillna(0), cluster_intensity.fillna(method='ffill')
    
    vol_persistence, vol_intensity = volume_clustering(data['volume'])
    data['vol_cluster_persistence'] = vol_persistence
    data['vol_cluster_intensity'] = vol_intensity
    
    # Price-Volume Fractal Correlation
    pv_correlation = data['close'].pct_change().rolling(window=5).corr(data['volume'].pct_change())
    data['fractal_sync'] = pv_correlation.fillna(method='ffill')
    
    # Intraday Microstructure Asymmetry
    # Bidirectional Pressure Analysis
    data['upward_pressure'] = ((data['high'] - data['open']) + (data['close'] - data['high'].shift(1))) / data['close']
    data['downward_pressure'] = ((data['open'] - data['low']) + (data['low'].shift(1) - data['close'])) / data['close']
    
    # Pressure imbalance
    data['pressure_imbalance'] = (data['upward_pressure'] - data['downward_pressure']) / (data['upward_pressure'] + data['downward_pressure'] + 1e-8)
    
    # Volume-weighted pressure signals
    volume_weight = data['volume'] / data['volume'].rolling(window=10).mean()
    data['volume_weighted_pressure'] = data['pressure_imbalance'] * volume_weight
    
    # Price-Volume Entropy Dynamics
    def price_entropy(price_series, window):
        """Calculate price entropy using price movements"""
        returns = price_series.pct_change().dropna()
        
        if len(returns) < window:
            return pd.Series([np.nan] * len(price_series), index=price_series.index)
        
        entropy_values = []
        for i in range(len(price_series)):
            if i < window:
                entropy_values.append(np.nan)
            else:
                window_returns = returns.iloc[i-window:i]
                # Calculate probability distribution of returns
                hist, _ = np.histogram(window_returns, bins=5, density=True)
                hist = hist[hist > 0]  # Remove zero probabilities
                entropy = -np.sum(hist * np.log(hist))
                entropy_values.append(entropy)
        
        return pd.Series(entropy_values, index=price_series.index)
    
    data['price_entropy_5d'] = price_entropy(data['close'], 5)
    data['price_entropy_10d'] = price_entropy(data['close'], 10)
    
    # Entropy momentum and divergence
    data['entropy_momentum'] = data['price_entropy_5d'] - data['price_entropy_10d']
    
    # Multi-Dimensional Momentum Coupling
    # Horizontal momentum (traditional price momentum)
    data['price_momentum_5d'] = data['close'].pct_change(5)
    data['price_momentum_10d'] = data['close'].pct_change(10)
    
    # Vertical momentum (range-based)
    data['range_momentum'] = (data['high'] - data['low']).pct_change(5)
    
    # Volume momentum
    data['volume_momentum'] = data['volume'].pct_change(5)
    
    # Dimension correlation analysis
    price_vol_corr = data['price_momentum_5d'].rolling(window=5).corr(data['volume_momentum'])
    data['momentum_coupling'] = price_vol_corr.fillna(method='ffill')
    
    # Combine all signals into final alpha factor
    # Weight different components based on their predictive power
    alpha = (
        0.25 * data['fractal_momentum'].fillna(0) +
        0.20 * data['volume_weighted_pressure'].fillna(0) +
        0.15 * data['entropy_momentum'].fillna(0) +
        0.20 * data['momentum_coupling'].fillna(0) +
        0.10 * data['vol_cluster_intensity'].fillna(0) +
        0.10 * data['fractal_sync'].fillna(0)
    )
    
    # Normalize the final alpha factor
    alpha_normalized = (alpha - alpha.rolling(window=20).mean()) / (alpha.rolling(window=20).std() + 1e-8)
    
    return alpha_normalized.fillna(method='ffill')
