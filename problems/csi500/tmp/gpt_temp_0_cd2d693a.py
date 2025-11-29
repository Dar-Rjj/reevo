import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor combining momentum regime transitions, 
    volume-price divergence, and opening auction effects
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Momentum Persistence Under Regime Shifts
    # Volatility regime classification
    data['vol_10d'] = data['close'].pct_change().rolling(window=10).std()
    data['vol_30d'] = data['close'].pct_change().rolling(window=30).std()
    data['vol_ratio'] = data['vol_10d'] / data['vol_30d']
    
    # Volatility regime flags
    data['high_vol_regime'] = ((data['vol_ratio'] > 1.2) & (data['vol_10d'] > data['vol_10d'].rolling(60).median())).astype(int)
    data['low_vol_regime'] = ((data['vol_ratio'] < 0.8) & (data['vol_10d'] < data['vol_10d'].rolling(60).median())).astype(int)
    
    # Trend regime identification
    data['ret_5d'] = data['close'].pct_change(5)
    data['ret_20d'] = data['close'].pct_change(20)
    data['trend_aligned'] = ((data['ret_5d'] > 0) & (data['ret_20d'] > 0)) | ((data['ret_5d'] < 0) & (data['ret_20d'] < 0))
    
    # Price level relative to moving ranges
    data['high_20d'] = data['high'].rolling(20).max()
    data['low_20d'] = data['low'].rolling(20).min()
    data['price_position'] = (data['close'] - data['low_20d']) / (data['high_20d'] - data['low_20d'])
    
    # Momentum persistence measurement
    data['ret_1d'] = data['close'].pct_change()
    data['ret_3d'] = data['close'].pct_change(3)
    
    # 1-day lag return correlation (simplified)
    data['momentum_persistence'] = data['ret_1d'].rolling(10).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 and not np.isnan(x).any() else 0
    )
    
    # Regime-specific momentum strength
    data['momentum_high_vol'] = data['ret_3d'] * data['high_vol_regime']
    data['momentum_low_vol'] = data['ret_3d'] * data['low_vol_regime'] * (1 + data['momentum_persistence'])
    
    # Transition-adaptive signal
    data['regime_transition'] = (data['high_vol_regime'].diff() != 0) | (data['low_vol_regime'].diff() != 0)
    data['momentum_signal'] = (
        data['ret_3d'] * (1 + 0.5 * data['regime_transition'].astype(int)) * 
        (1 + 0.3 * data['trend_aligned'].astype(int))
    )
    
    # 2. Volume-Price Divergence Acceleration
    # Volume momentum characteristics
    data['volume_3d_growth'] = data['volume'].pct_change(3)
    data['volume_acceleration'] = data['volume_3d_growth'] - data['volume_3d_growth'].shift(3)
    
    # Volume volatility relationship
    data['price_range'] = (data['high'] - data['low']) / data['close']
    data['volume_range'] = (data['volume'].rolling(5).max() - data['volume'].rolling(5).min()) / data['volume'].rolling(5).mean()
    data['vol_price_corr'] = data['price_range'].rolling(10).corr(data['volume_range'])
    
    # Price-volume divergence dynamics
    data['price_momentum'] = data['close'].pct_change(3)
    data['volume_momentum'] = data['volume'].pct_change(3)
    
    data['divergence'] = (
        (data['price_momentum'] - data['price_momentum'].rolling(10).mean()) - 
        (data['volume_momentum'] - data['volume_momentum'].rolling(10).mean())
    )
    
    # Divergence persistence
    data['divergence_persistence'] = data['divergence'].rolling(5).apply(
        lambda x: 1 if all(x > 0) else (-1 if all(x < 0) else 0)
    )
    
    # Acceleration divergence signals
    data['volume_divergence_signal'] = (
        data['divergence'] * (1 + 0.2 * data['divergence_persistence']) * 
        np.where(data['volume_acceleration'] > 0, 1.1, 0.9)
    )
    
    # 3. Opening Auction Imbalance Effects
    # Opening gap characteristics
    data['overnight_return'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['opening_gap_magnitude'] = data['overnight_return'].abs()
    
    # Opening volume concentration (simulated with available data)
    data['avg_volume_10d'] = data['volume'].rolling(10).mean()
    data['opening_volume_ratio'] = data['volume'] / data['avg_volume_10d']  # Approximation
    
    # Auction imbalance persistence
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['gap_absorption'] = data['overnight_return'] * data['intraday_return']
    
    # Enhanced opening signals
    data['opening_signal'] = (
        data['overnight_return'] * (1 - 0.5 * np.sign(data['overnight_return'] * data['intraday_return'])) *
        data['opening_volume_ratio']
    )
    
    # 4. Dynamic Factor Integration Framework
    # Regime-dependent signal weighting
    data['momentum_weight'] = np.where(data['high_vol_regime'] == 1, 0.4, 
                                     np.where(data['low_vol_regime'] == 1, 0.6, 0.5))
    
    data['volume_weight'] = np.where(data['high_vol_regime'] == 1, 0.3, 
                                   np.where(data['low_vol_regime'] == 1, 0.2, 0.25))
    
    data['opening_weight'] = np.where(data['high_vol_regime'] == 1, 0.3, 
                                    np.where(data['low_vol_regime'] == 1, 0.2, 0.25))
    
    # Multi-timeframe signal alignment
    data['signal_1d'] = data['ret_1d']
    data['signal_3d'] = data['ret_3d']
    data['timeframe_alignment'] = np.sign(data['signal_1d']) == np.sign(data['signal_3d'])
    
    # Final alpha construction
    data['combined_alpha'] = (
        data['momentum_weight'] * data['momentum_signal'] +
        data['volume_weight'] * data['volume_divergence_signal'] +
        data['opening_weight'] * data['opening_signal']
    ) * (1 + 0.2 * data['timeframe_alignment'].astype(int))
    
    # Return the final alpha factor series
    return data['combined_alpha']
