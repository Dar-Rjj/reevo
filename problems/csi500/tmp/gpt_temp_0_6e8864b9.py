import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Compression Momentum Divergence combined with other alpha factors
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Factor 1: Volatility Compression Momentum Divergence
    # Daily High-Low Range Reduction (5-day rolling)
    data['daily_range'] = data['high'] - data['low']
    data['range_ma5'] = data['daily_range'].rolling(window=5).mean()
    data['range_reduction'] = (data['range_ma5'] - data['daily_range']) / data['range_ma5']
    
    # Multi-day Range Contraction (3-day momentum)
    data['range_momentum'] = data['daily_range'].rolling(window=3).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
    )
    
    # Volume-Weighted Price Acceleration
    data['price_change'] = data['close'].pct_change()
    data['vol_weighted_momentum'] = (data['price_change'] * data['volume']).rolling(window=5).mean()
    data['momentum_accel'] = data['vol_weighted_momentum'].diff(3)
    
    # Compression-Momentum Divergence
    data['compression_divergence'] = data['range_reduction'] * data['momentum_accel']
    
    # Factor 2: Opening Imbalance Efficiency
    # Overnight Gap vs Opening Range
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['opening_range'] = (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()) / data['close'].shift(1)
    data['gap_vs_range'] = data['overnight_gap'] / data['opening_range'].replace(0, np.nan)
    
    # Opening Imbalance Persistence (first 30-min equivalent - using first hour high/low)
    data['early_high'] = data['high'].rolling(window=3).apply(lambda x: x.iloc[0])
    data['early_low'] = data['low'].rolling(window=3).apply(lambda x: x.iloc[0])
    data['early_range'] = (data['early_high'] - data['early_low']) / data['open']
    data['imbalance_persistence'] = data['gap_vs_range'] / data['early_range'].replace(0, np.nan)
    
    # Intraday Momentum Quality (smoothness)
    data['intraday_high_low'] = (data['high'] - data['low']) / data['open']
    data['close_to_open'] = abs(data['close'] - data['open']) / data['open']
    data['momentum_efficiency'] = data['close_to_open'] / data['intraday_high_low'].replace(0, np.nan)
    
    # Imbalance-Efficiency Signal
    data['imbalance_efficiency'] = data['imbalance_persistence'] * data['momentum_efficiency']
    
    # Factor 3: Price Level Memory Momentum Transfer
    # Recent pivot point significance (5-day high/low)
    data['recent_high'] = data['high'].rolling(window=5).max()
    data['recent_low'] = data['low'].rolling(window=5).min()
    data['distance_to_high'] = (data['close'] - data['recent_high']) / data['recent_high']
    data['distance_to_low'] = (data['close'] - data['recent_low']) / data['recent_low']
    
    # Level reaction strength
    data['level_reaction'] = np.where(
        abs(data['distance_to_high']) < 0.02, 
        -data['price_change'],  # Near resistance - negative reaction
        np.where(
            abs(data['distance_to_low']) < 0.02,
            data['price_change'],  # Near support - positive reaction
            0
        )
    )
    
    # Session Transition Momentum
    data['overnight_return'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['momentum_transfer'] = data['overnight_return'] * data['intraday_return']
    
    # Memory-Transfer Signal
    data['memory_transfer'] = data['level_reaction'] * data['momentum_transfer']
    
    # Factor 4: Fractal Volume-Price Coherence
    # Price movement coherence (hourly equivalent - using rolling 4-period windows)
    data['price_std_4'] = data['close'].rolling(window=4).std()
    data['price_std_8'] = data['close'].rolling(window=8).std()
    data['price_coherence'] = data['price_std_4'] / data['price_std_8'].replace(0, np.nan)
    
    # Volume-Price Alignment
    data['volume_ma5'] = data['volume'].rolling(window=5).mean()
    data['volume_spike'] = data['volume'] / data['volume_ma5']
    data['volume_price_alignment'] = data['volume_spike'] * data['price_change'].abs()
    
    # Multi-timeframe Momentum Consistency
    data['momentum_3d'] = data['close'].pct_change(3)
    data['momentum_5d'] = data['close'].pct_change(5)
    data['momentum_consistency'] = data['momentum_3d'] * data['momentum_5d']
    
    # Fractal Coherence Signal
    data['fractal_coherence'] = data['price_coherence'] * data['momentum_consistency']
    
    # Factor 5: Range Persistence Volume Confirmation
    # Range Persistence (5-day stability)
    data['range_std_5'] = data['daily_range'].rolling(window=5).std()
    data['range_mean_5'] = data['daily_range'].rolling(window=5).mean()
    data['range_persistence'] = data['range_std_5'] / data['range_mean_5'].replace(0, np.nan)
    
    # Range Efficiency
    data['close_to_high'] = (data['close'] - data['low']) / data['daily_range'].replace(0, np.nan)
    data['range_efficiency'] = 1 - abs(data['close_to_high'] - 0.5) * 2  # 1 = perfect efficiency
    
    # Volume Confirmation
    data['volume_momentum'] = data['volume'].pct_change(3)
    data['price_volume_corr'] = data['close'].rolling(window=5).corr(data['volume'])
    
    # Persistence-Confirmation Signal
    data['persistence_confirmation'] = data['range_persistence'] * data['price_volume_corr']
    
    # Combine all factors with equal weighting
    factors = [
        'compression_divergence',
        'imbalance_efficiency', 
        'memory_transfer',
        'fractal_coherence',
        'persistence_confirmation'
    ]
    
    # Normalize each factor by its rolling z-score (20-day window)
    combined_factor = pd.Series(0, index=data.index)
    for factor in factors:
        factor_series = data[factor].copy()
        # Remove outliers beyond 3 standard deviations
        mean = factor_series.rolling(window=20).mean()
        std = factor_series.rolling(window=20).std()
        z_score = (factor_series - mean) / std.replace(0, np.nan)
        z_score = z_score.clip(-3, 3)  # Winsorize
        combined_factor += z_score
    
    # Final factor series
    final_factor = combined_factor.rank(pct=True) - 0.5  # Center around 0
    
    return final_factor
