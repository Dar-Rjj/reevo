import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Transition Factor
    Identifies transitions between high and low volatility regimes and generates
    signals based on price behavior, volume dynamics, and regime characteristics.
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Basic calculations
    data['range'] = (data['high'] - data['low']) / data['close']
    data['close_prev'] = data['close'].shift(1)
    data['volume_prev'] = data['volume'].shift(1)
    data['price_change'] = data['close'] - data['close_prev']
    data['volume_change'] = data['volume'] - data['volume_prev']
    
    # Volatility State Identification
    data['vol_5d_avg'] = data['range'].rolling(window=5, min_periods=3).mean()
    data['vol_regime'] = np.where(data['range'] > data['vol_5d_avg'], 1, 
                                 np.where(data['range'] < data['vol_5d_avg'], -1, 0))
    
    # Volatility Momentum
    data['range_prev'] = data['range'].shift(1)
    data['vol_momentum'] = (data['range'] / data['range_prev'].replace(0, np.nan)) - 1
    data['vol_momentum'] = data['vol_momentum'].fillna(0)
    
    # Volatility Persistence
    data['regime_persistence'] = 0
    for i in range(1, len(data)):
        if data['vol_regime'].iloc[i] == data['vol_regime'].iloc[i-1]:
            data['regime_persistence'].iloc[i] = data['regime_persistence'].iloc[i-1] + 1
    
    # Multi-timeframe Volatility
    data['daily_vol'] = abs(data['close'] - data['close_prev'])
    data['intraday_vs_daily_vol'] = data['range'] / (data['daily_vol'].replace(0, np.nan) + 1e-8)
    data['intraday_vs_daily_vol'] = data['intraday_vs_daily_vol'].fillna(0)
    
    # Price Behavior - Trend Characteristics
    data['close_3d_ago'] = data['close'].shift(3)
    data['high_3d'] = data['high'].rolling(window=3, min_periods=2).max()
    data['low_3d'] = data['low'].rolling(window=3, min_periods=2).min()
    data['trend_strength'] = abs(data['close'] - data['close_3d_ago']) / (data['high_3d'] - data['low_3d'] + 1e-8)
    
    # Trend Persistence
    data['return_1d'] = data['close'] / data['close_prev'] - 1
    data['return_3d'] = data['close'] / data['close_3d_ago'] - 1
    data['trend_persistence'] = ((data['return_1d'] > 0) & (data['return_3d'] > 0)) | ((data['return_1d'] < 0) & (data['return_3d'] < 0))
    data['trend_persistence'] = data['trend_persistence'].astype(int)
    
    # Reversal Patterns
    data['high_5d'] = data['high'].rolling(window=5, min_periods=3).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=3).min()
    data['oversold_bought'] = (data['close'] - data['low_5d']) / (data['high_5d'] - data['low_5d'] + 1e-8)
    
    data['close_5d_ago'] = data['close'].shift(5)
    data['range_5d'] = data['high_5d'] - data['low_5d']
    data['mean_reversion'] = abs(data['close'] - data['close_5d_ago']) / (data['range_5d'] + 1e-8)
    
    # Breakout Behavior
    data['price_breakout'] = ((data['close'] > data['high_5d']) | (data['close'] < data['low_5d'])).astype(int)
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_breakout'] = (data['volume'] > data['volume_5d_avg']).astype(int)
    data['combined_breakout'] = data['price_breakout'] * data['volume_breakout']
    
    # Volume-Volatility Dynamics
    data['volume_efficiency'] = data['volume'] / ((data['high'] - data['low']) + 1e-8)
    data['volume_volatility'] = data['volume'] / data['volume_prev'].replace(0, np.nan) - 1
    data['volume_volatility'] = data['volume_volatility'].fillna(0)
    
    # Volume-Price Correlation
    data['directional_volume'] = np.sign(data['price_change']) * np.sign(data['volume_change'])
    data['volatility_change'] = data['range'] - data['range_prev']
    data['volume_vol_alignment'] = np.sign(data['volume_change']) * np.sign(data['volatility_change'])
    
    data['volume_20d_avg'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_intensity'] = data['volume'] / (data['volume_20d_avg'] + 1e-8)
    
    # Regime Transition Detection
    data['regime_change'] = data['vol_regime'].diff()
    data['high_to_low_transition'] = ((data['regime_change'] == -2) | (data['regime_change'] == -1)).astype(int)
    data['low_to_high_transition'] = ((data['regime_change'] == 2) | (data['regime_change'] == 1)).astype(int)
    
    # Transition strength
    data['vol_change_magnitude'] = abs(data['range'] - data['range_prev'])
    
    # Signal Generation
    # Volatility breakout signals
    data['vol_breakout_signal'] = (
        data['high_to_low_transition'] * -1 + 
        data['low_to_high_transition'] * 1
    ) * data['vol_change_magnitude']
    
    # Price pattern confirmation
    data['trend_confirmation'] = data['trend_persistence'] * data['trend_strength']
    data['reversal_signal'] = (0.5 - data['oversold_bought']) * data['mean_reversion']
    
    # Volume confirmation
    data['volume_spike_signal'] = data['volume_breakout'] * data['volume_intensity']
    data['volume_alignment_signal'] = data['volume_vol_alignment'] * data['directional_volume']
    
    # Composite Signal Construction
    # Multi-timeframe integration
    data['intraday_signal'] = data['intraday_vs_daily_vol'] * data['volume_efficiency']
    data['daily_signal'] = data['trend_confirmation'] + data['reversal_signal']
    
    # Regime-specific weighting
    high_vol_weight = 1.2
    low_vol_weight = 0.8
    transition_weight = 1.5
    
    data['regime_weight'] = np.where(data['vol_regime'] == 1, high_vol_weight,
                                   np.where(data['vol_regime'] == -1, low_vol_weight, transition_weight))
    
    # Dynamic signal adjustment
    data['transition_filter'] = (data['high_to_low_transition'] + data['low_to_high_transition']) * 2
    
    # Final composite factor
    data['composite_factor'] = (
        data['vol_breakout_signal'] * 0.3 +
        data['intraday_signal'] * 0.2 +
        data['daily_signal'] * 0.25 +
        data['volume_spike_signal'] * 0.15 +
        data['volume_alignment_signal'] * 0.1
    ) * data['regime_weight'] * (1 + data['transition_filter'])
    
    # Apply regime persistence smoothing
    persistence_smoothing = 1 / (1 + np.exp(-data['regime_persistence'] / 3))
    data['final_factor'] = data['composite_factor'] * persistence_smoothing
    
    # Clean up and return
    result = data['final_factor'].fillna(0)
    return result
