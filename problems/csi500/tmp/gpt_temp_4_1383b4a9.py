import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price and volume metrics
    data['returns'] = data['close'].pct_change()
    data['high_low_range'] = (data['high'] - data['low']) / data['close']
    data['open_close_range'] = abs(data['close'] - data['open']) / data['close']
    
    # Define midday point (assume 12:00 as rough midpoint)
    # Using high/low of first half vs second half as proxy
    data['morning_high'] = data['high'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else x[0])
    data['morning_low'] = data['low'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else x[0])
    
    # Morning session metrics (Open to Midday proxy)
    data['morning_price_change'] = (data['morning_high'] + data['morning_low']) / 2 - data['open']
    data['morning_price_dir'] = np.sign(data['morning_price_change'])
    
    # Afternoon session metrics (Midday to Close proxy)
    data['afternoon_price_change'] = data['close'] - (data['morning_high'] + data['morning_low']) / 2
    data['afternoon_price_dir'] = np.sign(data['afternoon_price_change'])
    
    # Volume metrics - using rolling averages as proxy for session volumes
    data['morning_volume_ratio'] = data['volume'].rolling(window=5).apply(lambda x: x[0] / np.mean(x[1:]) if len(x) == 5 else 1.0)
    data['afternoon_volume_ratio'] = data['volume'].rolling(window=5).apply(lambda x: x[-1] / np.mean(x[:-1]) if len(x) == 5 else 1.0)
    
    # Intraday Price-Volume Divergence
    # Morning divergence: price direction vs volume trend
    data['morning_divergence'] = 0
    mask_bullish_morning = (data['morning_price_dir'] > 0) & (data['morning_volume_ratio'] < 0.8)
    mask_bearish_morning = (data['morning_price_dir'] < 0) & (data['morning_volume_ratio'] > 1.2)
    data.loc[mask_bullish_morning, 'morning_divergence'] = -1  # Price up, volume down
    data.loc[mask_bearish_morning, 'morning_divergence'] = 1   # Price down, volume up
    
    # Afternoon divergence
    data['afternoon_divergence'] = 0
    mask_bullish_afternoon = (data['afternoon_price_dir'] > 0) & (data['afternoon_volume_ratio'] < 0.8)
    mask_bearish_afternoon = (data['afternoon_price_dir'] < 0) & (data['afternoon_volume_ratio'] > 1.2)
    data.loc[mask_bullish_afternoon, 'afternoon_divergence'] = -1  # Price up, volume down
    data.loc[mask_bearish_afternoon, 'afternoon_divergence'] = 1   # Price down, volume up
    
    # Multi-timeframe divergence patterns
    data['short_term_trend'] = data['close'].pct_change(periods=3).rolling(window=5).mean()
    data['medium_term_trend'] = data['close'].pct_change(periods=10).rolling(window=10).mean()
    
    # Divergence persistence
    data['divergence_persistence'] = (
        data['morning_divergence'].rolling(window=3).sum() + 
        data['afternoon_divergence'].rolling(window=3).sum()
    )
    
    # Trading Efficiency Metrics
    # Intraday Range Utilization
    data['range_utilization'] = data['open_close_range'] / (data['high_low_range'] + 1e-8)
    
    # Price Path Efficiency (Actual vs Minimum Distance)
    min_distance = abs(data['close'] - data['open'])
    actual_distance = (
        abs(data['high'] - data['open']) + 
        abs(data['low'] - data['open']) + 
        abs(data['close'] - data['open'])
    ) / 3
    data['price_efficiency'] = min_distance / (actual_distance + 1e-8)
    
    # Volume Distribution Patterns
    # Volume concentration (using high-low range as proxy for key zones)
    data['volume_concentration'] = data['volume'] / (data['high_low_range'] * data['close'] + 1e-8)
    
    # Volume profile skewness (using price-volume relationship)
    price_volume_corr = data['close'].rolling(window=10).corr(data['volume'])
    data['volume_skewness'] = price_volume_corr.rolling(window=5).skew()
    
    # Generate Composite Alpha Factor
    # Divergence-based signals
    data['bullish_divergence'] = (
        ((data['morning_divergence'] > 0) | (data['afternoon_divergence'] > 0)) & 
        (data['medium_term_trend'] > 0)
    ).astype(int)
    
    data['bearish_divergence'] = (
        ((data['morning_divergence'] < 0) | (data['afternoon_divergence'] < 0)) & 
        (data['medium_term_trend'] < 0)
    ).astype(int)
    
    data['convergence_signal'] = (
        (data['morning_divergence'] == 0) & 
        (data['afternoon_divergence'] == 0) & 
        (abs(data['short_term_trend']) > 0.01)
    ).astype(int)
    
    # Base divergence signal
    data['divergence_signal'] = (
        data['bullish_divergence'] - data['bearish_divergence'] + 
        0.5 * data['convergence_signal'] * np.sign(data['short_term_trend'])
    )
    
    # Efficiency-weighted signal enhancement
    efficiency_score = (
        0.4 * data['range_utilization'].rolling(window=5).mean() +
        0.4 * data['price_efficiency'].rolling(window=5).mean() +
        0.2 * (1 - abs(data['volume_skewness'].rolling(window=5).mean()))
    )
    
    # Normalize efficiency score
    efficiency_zscore = (efficiency_score - efficiency_score.rolling(window=20).mean()) / (efficiency_score.rolling(window=20).std() + 1e-8)
    
    # Apply efficiency weighting
    efficiency_weight = 1 + 0.5 * np.tanh(efficiency_zscore / 2)  # 0.5 to 1.5 range
    
    # Final composite factor
    data['composite_factor'] = (
        data['divergence_signal'] * efficiency_weight +
        0.3 * data['divergence_persistence'] +
        0.2 * data['volume_concentration'].pct_change(periods=3)
    )
    
    # Clean and return
    factor = data['composite_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor
