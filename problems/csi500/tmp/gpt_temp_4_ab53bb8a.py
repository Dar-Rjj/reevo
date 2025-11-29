import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Gap Momentum with Volume-Price Divergence Synthesis alpha factor
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['overnight_gap'] = data['open'] - data['close'].shift(1)
    data['high_low_range'] = data['high'] - data['low']
    data['gap_absorption'] = (data['high'] - data['open']) - (data['open'] - data['low'])
    data['gap_momentum_ratio'] = data['overnight_gap'] / data['high_low_range']
    
    # Gap Persistence
    data['gap_retention'] = (data['close'] - data['open']) / data['overnight_gap']
    data['gap_directional_consistency'] = np.sign(data['overnight_gap']) * np.sign(data['close'] - data['open'])
    
    # Volume-Price Divergence Analysis
    data['price_trend_3d'] = data['close'] - data['close'].shift(3)
    data['volume_trend_3d'] = data['volume'] - data['volume'].shift(3)
    data['volume_price_divergence'] = data['price_trend_3d'] * data['volume_trend_3d'] * -1
    
    # Accumulation/Distribution Detection
    data['vw_price_position'] = ((data['close'] - data['low']) / data['high_low_range']) * data['volume']
    data['accumulation_change'] = data['vw_price_position'] - data['vw_price_position'].shift(2)
    data['accumulation_gap_confirmation'] = data['accumulation_change'] * data['gap_momentum_ratio']
    
    # Multi-timeframe Momentum Reversal
    data['range_extension_2d'] = data['high_low_range'] / (data['high'].shift(2) - data['low'].shift(2))
    data['momentum_5d'] = data['close'] - data['close'].shift(5)
    data['momentum_10d'] = data['close'] - data['close'].shift(10)
    
    # True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_5d'] = data['true_range'].rolling(window=5).mean()
    data['tr_ratio'] = data['true_range'] / data['atr_5d']
    
    # Short-term reversal component
    reversal_threshold = 1.2
    data['reversal_component'] = np.where(data['range_extension_2d'] > reversal_threshold, 
                                         -1 * np.sign(data['momentum_5d']), 1.0)
    
    # Momentum convergence/divergence
    data['momentum_convergence'] = np.sign(data['momentum_5d']) * np.sign(data['momentum_10d'])
    
    # Volatility-adjusted reversal timing
    data['volatility_reversal'] = np.where(data['tr_ratio'] > 1.5, 
                                          data['reversal_component'] * data['tr_ratio'], 
                                          data['reversal_component'])
    
    # Simplified intraday pattern recognition (using close-open as proxy)
    data['intraday_range'] = data['high'] - data['low']
    data['intraday_move'] = data['close'] - data['open']
    data['intraday_strength'] = data['intraday_move'] / data['intraday_range']
    
    # Combine core components
    data['gap_momentum_strength'] = (data['gap_momentum_ratio'] * 
                                    data['gap_directional_consistency'] * 
                                    np.where(abs(data['gap_retention']) < 10, data['gap_retention'], 0))
    
    # Core divergence signal
    data['core_divergence'] = (data['gap_momentum_strength'] * 
                              data['volume_price_divergence'] * 
                              data['accumulation_gap_confirmation'])
    
    # Apply multi-timeframe reversal filter
    data['filtered_divergence'] = data['core_divergence'] * data['volatility_reversal']
    
    # Apply intraday pattern strength
    data['final_alpha'] = (data['filtered_divergence'] * 
                          data['intraday_strength'] * 
                          data['momentum_convergence'])
    
    # Handle infinite values and extreme outliers
    data['final_alpha'] = data['final_alpha'].replace([np.inf, -np.inf], np.nan)
    
    # Cross-sectional normalization (z-score within each day)
    def cross_sectional_zscore(series):
        if len(series.dropna()) > 1:
            return (series - series.mean()) / series.std()
        else:
            return series
    
    # Apply cross-sectional normalization
    alpha_series = data.groupby(data.index)['final_alpha'].transform(cross_sectional_zscore)
    
    return alpha_series
