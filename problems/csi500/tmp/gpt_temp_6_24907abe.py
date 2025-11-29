import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns and previous close
    data['prev_close'] = data.groupby(level=1)['close'].shift(1)
    data['daily_return'] = (data['close'] - data['prev_close']) / data['prev_close']
    
    # Volatility-Scaled Intraday Momentum Divergence
    # Intraday momentum components
    data['range_momentum'] = (data['high'] - data['low']) / data['prev_close']
    data['gap_momentum'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Momentum divergence
    data['momentum_divergence'] = data['range_momentum'] - data['gap_momentum']
    
    # True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # 10-day average True Range
    data['atr_10'] = data.groupby(level=1)['true_range'].transform(
        lambda x: x.rolling(window=10, min_periods=5).mean()
    )
    
    # Volatility scaling
    data['vol_scaled_momentum'] = data['momentum_divergence'] / data['atr_10']
    
    # Range-Constrained Volume Gap Dynamics
    # Gap filling dynamics
    denominator = abs(data['open'] - data['prev_close'])
    data['gap_filling'] = np.where(
        denominator > 0, 
        (data['close'] - data['open']) / denominator, 
        0
    )
    
    # Volume confirmation
    data['volume_ma_20'] = data.groupby(level=1)['volume'].transform(
        lambda x: x.rolling(window=20, min_periods=10).mean()
    )
    data['volume_confirmation'] = data['volume'] / data['volume_ma_20']
    
    # Range adjustment
    price_range = data['high'] - data['low']
    data['range_constrained_volume'] = np.where(
        price_range > 0,
        data['gap_filling'] * data['volume_confirmation'] / price_range,
        0
    )
    
    # Multi-Timeframe Gap Persistence
    # Overnight gap
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Gap persistence ratio
    denominator_gap = abs(data['overnight_gap'])
    data['gap_persistence'] = np.where(
        denominator_gap > 1e-8,  # Avoid division by very small numbers
        data['daily_return'] / data['overnight_gap'],
        0
    )
    
    # Volatility context
    data['volatility_20d'] = data.groupby(level=1)['daily_return'].transform(
        lambda x: x.rolling(window=20, min_periods=10).std()
    )
    
    # Volume-Efficient Momentum Convergence
    # Momentum convergence
    data['return_5d'] = data.groupby(level=1)['close'].transform(
        lambda x: x.pct_change(periods=5)
    )
    data['return_20d'] = data.groupby(level=1)['close'].transform(
        lambda x: x.pct_change(periods=20)
    )
    
    denominator_momentum = abs(data['return_20d'])
    data['momentum_convergence'] = np.where(
        denominator_momentum > 1e-8,
        data['return_5d'] / data['return_20d'],
        0
    )
    
    # Volume efficiency
    data['volume_efficiency'] = np.where(
        data['volume'] > 0,
        data['daily_return'] / data['volume'],
        0
    )
    
    # Price context - 52-week high/low
    data['high_52w'] = data.groupby(level=1)['high'].transform(
        lambda x: x.rolling(window=252, min_periods=126).max()
    )
    data['low_52w'] = data.groupby(level=1)['low'].transform(
        lambda x: x.rolling(window=252, min_periods=126).min()
    )
    
    denominator_price = data['high_52w'] - data['low_52w']
    data['price_context'] = np.where(
        denominator_price > 0,
        (data['close'] - data['low_52w']) / denominator_price,
        0.5
    )
    
    # Combine all factors with equal weights
    factor = (
        data['vol_scaled_momentum'] + 
        data['range_constrained_volume'] + 
        (data['gap_persistence'] / (data['volatility_20d'] + 1e-8)) + 
        data['momentum_convergence'] * data['volume_efficiency'] * data['price_context']
    )
    
    # Return factor series indexed by date
    return factor
