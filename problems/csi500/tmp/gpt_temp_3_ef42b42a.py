import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum Fractality factor combining price path complexity, 
    volume clustering, and momentum persistence analysis.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns and ranges
    data['returns'] = data['close'].pct_change()
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['intraday_volatility'] = (data['high'] - data['low']) / data['open']
    
    # Fractal Dimension Calculation - Price path complexity
    # Using Hurst exponent approximation via rescaled range analysis
    def calculate_hurst(series, window=20):
        """Calculate Hurst exponent as proxy for fractal dimension"""
        hurst_values = []
        for i in range(len(series)):
            if i < window:
                hurst_values.append(np.nan)
                continue
                
            window_data = series.iloc[i-window:i]
            if len(window_data) < 2:
                hurst_values.append(np.nan)
                continue
                
            # Calculate rescaled range
            mean_val = window_data.mean()
            deviations = window_data - mean_val
            cumulative_deviations = deviations.cumsum()
            range_val = cumulative_deviations.max() - cumulative_deviations.min()
            std_val = window_data.std()
            
            if std_val == 0:
                hurst_values.append(np.nan)
            else:
                hurst = np.log(range_val / std_val) / np.log(window)
                hurst_values.append(hurst)
        
        return pd.Series(hurst_values, index=series.index)
    
    # Calculate price fractal dimension
    data['price_hurst'] = calculate_hurst(data['close'], window=20)
    
    # Volume clustering patterns
    def volume_clustering(volume_series, window=10):
        """Calculate volume clustering using entropy"""
        clustering_values = []
        for i in range(len(volume_series)):
            if i < window:
                clustering_values.append(np.nan)
                continue
                
            window_data = volume_series.iloc[i-window:i]
            if window_data.sum() == 0:
                clustering_values.append(np.nan)
                continue
                
            # Normalize volume
            normalized_vol = window_data / window_data.sum()
            # Calculate entropy (lower entropy = more clustering)
            entropy = -np.sum(normalized_vol * np.log(normalized_vol + 1e-10))
            clustering_values.append(entropy)
        
        return pd.Series(clustering_values, index=volume_series.index)
    
    data['volume_entropy'] = volume_clustering(data['volume'], window=10)
    
    # Momentum Persistence Analysis
    # Short-term vs medium-term momentum alignment
    data['momentum_5d'] = data['close'].pct_change(5)
    data['momentum_10d'] = data['close'].pct_change(10)
    data['momentum_alignment'] = np.sign(data['momentum_5d']) * np.sign(data['momentum_10d'])
    
    # Fractal breakdown detection - sudden dimension changes
    data['hurst_change'] = data['price_hurst'].diff(3)
    data['hurst_volatility'] = data['price_hurst'].rolling(window=10, min_periods=5).std()
    
    # Opening Volatility Cascade
    # Pre-open volatility estimation using previous session's closing structure
    data['prev_close_vol'] = data['close'].pct_change().rolling(window=5, min_periods=3).std()
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Opening volatility propagation
    data['opening_range'] = (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()) / data['open']
    data['volatility_transmission'] = data['prev_close_vol'] / (data['intraday_volatility'] + 1e-10)
    
    # Price-Volume Asymmetry Detection
    # Directional volume asymmetry using close-to-close returns
    data['up_volume'] = np.where(data['returns'] > 0, data['volume'], 0)
    data['down_volume'] = np.where(data['returns'] < 0, data['volume'], 0)
    data['volume_asymmetry'] = (data['up_volume'].rolling(window=10).sum() - 
                               data['down_volume'].rolling(window=10).sum()) / data['volume'].rolling(window=10).sum()
    
    # Temporal volume asymmetry - early vs late session
    # Using first vs last hour approximation (assuming data is daily)
    data['volume_acceleration'] = data['volume'].pct_change(3)
    
    # Range Expansion Thermodynamics
    # Energy accumulation - price compression
    data['range_compression'] = data['daily_range'].rolling(window=10).std()
    data['compression_ratio'] = data['daily_range'] / data['daily_range'].rolling(window=20).mean()
    
    # Energy release - expansion velocity
    data['expansion_velocity'] = data['daily_range'].diff(3)
    
    # Combine factors for final heuristic
    # High fractal dimension + volume clustering → Chaotic momentum (negative)
    chaotic_momentum = data['price_hurst'] * (1 - data['volume_entropy'].rank(pct=True))
    
    # Momentum persistence alignment
    momentum_strength = data['momentum_alignment'] * (data['momentum_5d'].abs() + data['momentum_10d'].abs())
    
    # Fractal regime stability
    regime_stability = -data['hurst_change'].abs() - data['hurst_volatility']
    
    # Volatility cascade efficiency
    cascade_efficiency = -data['volatility_transmission'].abs() * data['overnight_gap'].abs()
    
    # Volume asymmetry strength
    volume_strength = data['volume_asymmetry'].abs() * data['volume_acceleration']
    
    # Energy accumulation/release cycle
    energy_cycle = data['compression_ratio'] * data['expansion_velocity']
    
    # Final composite factor
    factor = (
        0.3 * chaotic_momentum.rank(pct=True) +
        0.25 * momentum_strength.rank(pct=True) +
        0.2 * regime_stability.rank(pct=True) +
        0.15 * cascade_efficiency.rank(pct=True) +
        0.1 * volume_strength.rank(pct=True)
    )
    
    # Remove any remaining NaN values
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return factor
