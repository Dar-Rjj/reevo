import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Session Momentum-Volatility Regime Alpha Factor
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Calculate basic price features
    data['prev_close'] = data['close'].shift(1)
    data['daily_range'] = (data['high'] - data['low']) / data['prev_close']
    
    # Morning session features (assuming 10:30 data available)
    data['morning_high'] = data['high'].rolling(window=30, min_periods=1).max()
    data['morning_low'] = data['low'].rolling(window=30, min_periods=1).min()
    data['morning_close'] = data['close'].rolling(window=30, min_periods=1).apply(lambda x: x[-1] if len(x) == 30 else np.nan)
    
    # Afternoon session features (assuming 15:00 data available)
    data['afternoon_open'] = data['open'].rolling(window=30, min_periods=1).apply(lambda x: x[0] if len(x) == 30 else np.nan)
    data['afternoon_high'] = data['high'].rolling(window=30, min_periods=1).max()
    data['afternoon_low'] = data['low'].rolling(window=30, min_periods=1).min()
    
    # Early Session Momentum Strength
    data['morning_momentum'] = (data['morning_close'] - data['open']) / (data['morning_high'] - data['morning_low'] + 1e-8)
    data['morning_momentum_persistence'] = data['morning_momentum'].rolling(window=3, min_periods=1).std()
    
    # Late Session Momentum Reversal
    data['afternoon_momentum'] = (data['close'] - data['afternoon_open']) / (data['afternoon_high'] - data['afternoon_low'] + 1e-8)
    data['momentum_decay'] = data['afternoon_momentum'] - data['morning_momentum']
    
    # Intraday Volatility Phase Analysis
    data['morning_volatility'] = (data['morning_high'] - data['morning_low']) / data['open']
    data['afternoon_volatility'] = (data['high'] - data['afternoon_low']) / data['afternoon_open']
    data['volatility_ratio'] = data['afternoon_volatility'] / (data['morning_volatility'] + 1e-8)
    
    # Volatility Regime Persistence
    data['volatility_regime'] = data['daily_range'].rolling(window=5, min_periods=1).mean()
    data['volatility_clustering'] = data['daily_range'].rolling(window=3, min_periods=1).std() / (data['daily_range'].rolling(window=3, min_periods=1).mean() + 1e-8)
    
    # Price Range Compression-Expansion Dynamics
    data['range_percentile'] = data['daily_range'].rolling(window=3, min_periods=1).apply(lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x) + 1e-8))
    data['range_expansion'] = data['daily_range'] / data['daily_range'].shift(1)
    
    # Volume-Volatility Divergence Patterns
    data['volume_volatility_ratio'] = data['volume'] / (data['daily_range'] + 1e-8)
    data['volume_concentration'] = data['volume_volatility_ratio'].rolling(window=3, min_periods=1).std()
    data['volatility_volume_confirmation'] = (data['daily_range'] > data['daily_range'].shift(1)) & (data['volume'] > data['volume'].shift(1))
    
    # Momentum-Volatility Regime Alignment
    data['high_vol_momentum'] = data['morning_momentum'] * (data['volatility_regime'] > data['volatility_regime'].rolling(window=5, min_periods=1).quantile(0.7))
    data['low_vol_breakout'] = data['afternoon_momentum'] * (data['volatility_regime'] < data['volatility_regime'].rolling(window=5, min_periods=1).quantile(0.3))
    
    # Cross-Session Regime Transition Signals
    data['regime_consistency'] = (data['morning_volatility'] > data['morning_volatility'].shift(1)) == (data['afternoon_volatility'] > data['afternoon_volatility'].shift(1))
    data['momentum_carryover'] = data['morning_momentum'] * data['afternoon_momentum']
    
    # Price Level Anchoring Dynamics
    data['open_anchor'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['price_clustering'] = data['close'].rolling(window=3, min_periods=1).std() / (data['close'].rolling(window=3, min_periods=1).mean() + 1e-8)
    
    # Composite Alpha Calculation
    alpha_components = []
    
    # Weighted momentum by volatility regime
    momentum_weighted = (data['high_vol_momentum'].fillna(0) * 0.6 + 
                        data['low_vol_breakout'].fillna(0) * 0.4)
    alpha_components.append(momentum_weighted)
    
    # Range expansion confirmation
    range_signal = data['range_expansion'].fillna(1) * data['range_percentile'].fillna(0.5)
    alpha_components.append(range_signal)
    
    # Volume-volatility alignment
    volume_signal = data['volume_concentration'].fillna(0) * data['volatility_volume_confirmation'].fillna(0)
    alpha_components.append(volume_signal)
    
    # Regime transition timing
    regime_signal = data['regime_consistency'].fillna(0) * data['momentum_carryover'].fillna(0)
    alpha_components.append(regime_signal)
    
    # Price anchoring dynamics
    anchor_signal = -data['open_anchor'].fillna(0) * data['price_clustering'].fillna(1)
    alpha_components.append(anchor_signal)
    
    # Combine all components with weights
    weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    composite_alpha = sum(w * comp for w, comp in zip(weights, alpha_components))
    
    # Final alpha series
    alpha_series = pd.Series(composite_alpha, index=data.index)
    
    return alpha_series
