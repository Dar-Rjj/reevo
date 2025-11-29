import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Trend Components
    # Trend strength
    data['trend_strength'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['trend_strength'] = data['trend_strength'].replace([np.inf, -np.inf], np.nan)
    
    # Trend exhaustion
    condition = data['close'] > data['open']
    data['trend_exhaustion'] = np.where(
        condition,
        (data['high'] - data['close']) / (data['close'] - data['low']),
        (data['close'] - data['low']) / (data['high'] - data['close'])
    )
    data['trend_exhaustion'] = data['trend_exhaustion'].replace([np.inf, -np.inf], np.nan)
    
    # Price efficiency
    data['price_efficiency'] = data['trend_strength'] * data['volume']
    
    # Liquidity Dynamics
    # Liquidity acceleration
    volume_median = data['volume'].rolling(window=5, min_periods=3).median()
    volume_std = data['volume'].rolling(window=5, min_periods=3).std()
    data['liquidity_acceleration'] = (data['volume'] - volume_median) / volume_std
    data['liquidity_acceleration'] = data['liquidity_acceleration'].replace([np.inf, -np.inf], np.nan)
    
    # Liquidity persistence
    volume_change = data['volume'].diff()
    data['liquidity_persistence'] = volume_change.rolling(window=10, min_periods=5).sum()
    
    # Bid-ask pressure
    data['bid_ask_pressure'] = ((2 * data['close'] - data['high'] - data['low']) / 
                               (data['high'] - data['low'])) * data['volume']
    data['bid_ask_pressure'] = data['bid_ask_pressure'].replace([np.inf, -np.inf], np.nan)
    
    # Market Regime Adaptation
    # Volatility regime
    daily_range = (data['high'] - data['low']) / data['open']
    data['volatility_regime'] = daily_range.rolling(window=20, min_periods=10).apply(lambda x: x.max() - x.min())
    
    # Trend regime
    data['trend_regime'] = data['open'].rolling(window=10, min_periods=5).corr(data['close'])
    
    # Regime-adjusted exhaustion
    data['regime_adjusted_exhaustion'] = data['trend_exhaustion'] * (1 + data['volatility_regime'])
    
    # Factor Synthesis
    # Combine trend exhaustion with liquidity acceleration
    trend_liquidity_component = data['regime_adjusted_exhaustion'] * data['liquidity_acceleration']
    
    # Adjust for market regime conditions
    regime_adjusted = trend_liquidity_component * (1 + data['trend_regime'].fillna(0))
    
    # Scale by price efficiency and bid-ask pressure
    scaling_factor = data['price_efficiency'] * data['bid_ask_pressure']
    final_factor = regime_adjusted * scaling_factor
    
    # Handle any remaining infinite values
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    
    return final_factor
