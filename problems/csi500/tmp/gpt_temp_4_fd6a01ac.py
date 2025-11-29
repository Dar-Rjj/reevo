import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price features
    data['open'] = df['open']
    data['high'] = df['high']
    data['low'] = df['low']
    data['close'] = df['close']
    data['volume'] = df['volume']
    data['amount'] = df['amount']
    
    # 1. Intraday Volatility Structure
    # Opening volatility dominance
    data['high_open_diff'] = data['high'] - data['open']
    data['open_low_diff'] = data['open'] - data['low']
    data['opening_asymmetry'] = data['high_open_diff'] - data['open_low_diff']
    
    # Closing volatility persistence
    data['close_low_diff'] = data['close'] - data['low']
    data['high_close_diff'] = data['high'] - data['close']
    data['closing_imbalance'] = data['close_low_diff'] - data['high_close_diff']
    
    # Daily range and volatility regime
    data['daily_range'] = data['high'] - data['low']
    data['range_pct'] = data['daily_range'] / data['close']
    
    # Volatility regime classification (rolling 20-day median as threshold)
    vol_threshold = data['range_pct'].rolling(window=20, min_periods=10).median()
    data['high_vol_regime'] = (data['range_pct'] > vol_threshold).astype(int)
    
    # 2. Price-Volume Fractal Dynamics
    # Volume clustering patterns
    data['volume_ma5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_cluster'] = (data['volume'] > data['volume_ma5']).rolling(window=3, min_periods=2).sum()
    
    # Price fractal dimension approximation (Hurst exponent-like)
    data['log_range'] = np.log(data['daily_range'])
    data['log_range_std'] = data['log_range'].rolling(window=10, min_periods=5).std()
    data['price_fractal'] = 2 - data['log_range_std'] / data['log_range_std'].rolling(window=20, min_periods=10).mean()
    
    # Volume-price fractal alignment
    data['volume_rank'] = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['range_rank'] = data['daily_range'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['fractal_alignment'] = 1 - abs(data['volume_rank'] - data['range_rank'])
    
    # 3. Momentum Regime Transitions
    # Opening gap momentum persistence
    data['prev_close'] = data['close'].shift(1)
    data['open_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['intraday_trend'] = (data['close'] - data['open']) / data['open']
    data['gap_momentum_alignment'] = np.sign(data['open_gap']) * np.sign(data['intraday_trend'])
    
    # Volatility expansion momentum
    data['range_expansion'] = data['daily_range'] / data['daily_range'].rolling(window=5, min_periods=3).mean()
    data['price_continuation'] = data['close'].pct_change().rolling(window=3, min_periods=2).sum()
    data['vol_expansion_momentum'] = data['range_expansion'] * data['price_continuation']
    
    # Regime transition momentum
    data['vol_regime_change'] = data['high_vol_regime'].diff()
    low_to_high_mask = data['vol_regime_change'] == 1
    data['regime_transition_momentum'] = 0
    data.loc[low_to_high_mask, 'regime_transition_momentum'] = data.loc[low_to_high_mask, 'price_continuation']
    
    # 4. Amount Flow Regime Analysis
    # Intraday amount distribution approximation
    data['amount_per_volume'] = data['amount'] / (data['volume'] + 1e-8)
    data['amount_concentration'] = data['amount_per_volume'].rolling(window=10, min_periods=5).std()
    
    # Amount volatility relationship
    data['amount_rank'] = data['amount'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['amount_vol_relationship'] = data['amount_rank'] * data['range_pct']
    
    # Regime-specific amount efficiency
    data['price_movement'] = abs(data['close'] - data['open']) / data['open']
    data['amount_efficiency'] = data['price_movement'] / (data['amount'] + 1e-8)
    
    # Different efficiency measures for volatility regimes
    data['amount_efficiency_low_vol'] = data['amount_efficiency'].copy()
    data['amount_efficiency_high_vol'] = data['amount_efficiency'].copy()
    data.loc[data['high_vol_regime'] == 0, 'amount_efficiency_high_vol'] = np.nan
    data.loc[data['high_vol_regime'] == 1, 'amount_efficiency_low_vol'] = np.nan
    
    # 5. Composite Signal Generation
    # Volatility regime momentum score
    data['vol_regime_momentum'] = (
        data['opening_asymmetry'].rolling(window=5, min_periods=3).mean() +
        data['closing_imbalance'].rolling(window=5, min_periods=3).mean() +
        data['regime_transition_momentum'].rolling(window=5, min_periods=3).mean()
    )
    
    # Enhanced by fractal alignment strength
    data['fractal_strength'] = data['fractal_alignment'] * data['price_fractal']
    
    # Confirmed by amount flow regime support
    data['amount_flow_support'] = (
        data['amount_vol_relationship'].rolling(window=5, min_periods=3).mean() -
        data['amount_efficiency'].rolling(window=10, min_periods=5).mean()
    )
    
    # Filtered by regime transition probability
    regime_transition_prob = data['vol_regime_change'].abs().rolling(window=20, min_periods=10).mean()
    data['transition_filter'] = 1 - regime_transition_prob
    
    # Final composite factor
    data['composite_factor'] = (
        data['vol_regime_momentum'] * 
        data['fractal_strength'] * 
        data['amount_flow_support'] * 
        data['transition_filter']
    )
    
    # Normalize the final factor
    factor = data['composite_factor'].copy()
    factor_ma = factor.rolling(window=20, min_periods=10).mean()
    factor_std = factor.rolling(window=20, min_periods=10).std()
    normalized_factor = (factor - factor_ma) / (factor_std + 1e-8)
    
    return normalized_factor
