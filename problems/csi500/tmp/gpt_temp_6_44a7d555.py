import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price metrics
    data['prev_close'] = data['close'].shift(1)
    data['daily_range'] = data['high'] - data['low']
    data['opening_gap'] = (data['open'] / data['prev_close']) - 1
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    
    # 1. Intraday Range Persistence
    data['range_efficiency'] = np.abs(data['close'] - data['open']) / (data['daily_range'] + 1e-8)
    
    # Gap survival persistence (3-day window)
    data['gap_survival'] = data['opening_gap'].rolling(window=3).apply(
        lambda x: np.mean(np.sign(x.iloc[0]) == np.sign(x)) if len(x) == 3 else np.nan
    )
    
    # 2. Volatility-Adjusted Momentum with Reversal Detection
    data['momentum_5d'] = data['close'].pct_change(5)
    data['momentum_20d'] = data['close'].pct_change(20)
    data['momentum_divergence'] = data['momentum_5d'] - data['momentum_20d']
    
    # Asymmetric reversal ratios
    data['up_reversal_ratio'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    data['down_reversal_ratio'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['reversal_strength'] = np.abs(data['up_reversal_ratio'] - data['down_reversal_ratio'])
    
    # 3. Volume Acceleration and Efficiency Analysis
    data['volume_ma_10d'] = data['volume'].rolling(window=10).mean()
    data['amount_ma_10d'] = data['amount'].rolling(window=10).mean()
    
    data['volume_acceleration'] = data['volume'] / data['volume_ma_10d'] - 1
    data['amount_deviation'] = data['amount'] / data['amount_ma_10d']
    
    data['price_volume_efficiency'] = data['intraday_return'] / (data['amount_deviation'] + 1e-8)
    
    # 4. Range Compression-Expansion Cycle Monitoring
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    
    # Range compression score (5-day window)
    data['range_ma_5d'] = data['true_range'].rolling(window=5).mean()
    data['range_ma_20d'] = data['true_range'].rolling(window=20).mean()
    data['compression_score'] = data['range_ma_5d'] / data['range_ma_20d']
    
    # Expansion breakout signals
    data['range_expansion'] = data['true_range'] / data['range_ma_5d'] - 1
    data['breakout_signal'] = data['range_expansion'] * np.sign(data['intraday_return'])
    
    # 5. Generate Composite Alpha Factor
    # Combine range persistence with momentum divergence
    range_momentum_component = data['range_efficiency'] * data['momentum_divergence'] * (1 + data['reversal_strength'])
    
    # Apply volume confirmation
    volume_weighted_component = range_momentum_component * (1 + data['volume_acceleration']) * np.abs(data['price_volume_efficiency'])
    
    # Incorporate range cycle dynamics
    expansion_boost = np.where(data['compression_score'] < 0.8, 1.5, 1.0)  # Boost during expansion
    compression_suppress = np.where(data['compression_score'] > 1.2, 0.5, 1.0)  # Suppress during compression
    
    regime_adjustment = expansion_boost * compression_suppress
    
    # Final composite factor
    composite_factor = volume_weighted_component * regime_adjustment * data['breakout_signal']
    
    # Clean and return the factor series
    factor_series = composite_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor_series
