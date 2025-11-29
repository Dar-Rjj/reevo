import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor mining function
    Input: DataFrame with open, high, low, close, amount, volume
    Output: Factor values as pandas Series
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor components
    factors = pd.DataFrame(index=data.index)
    
    # 1. Intraday Price Efficiency
    # Opening Gap vs Range Ratio
    factors['gap_range_ratio'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    # Price Path Optimality - measure of how efficiently price moves from open to close
    actual_distance = abs(data['close'] - data['open'])
    max_possible_distance = data['high'] - data['low']
    factors['price_path_optimality'] = actual_distance / max_possible_distance.where(max_possible_distance > 0, np.nan)
    
    # 2. Volume-Price Dynamics
    # Amount Impact Deviation
    price_change = data['close'].pct_change()
    amount_change = data['amount'].pct_change()
    factors['amount_impact_dev'] = price_change / amount_change.where(amount_change != 0, np.nan)
    
    # Volume Elasticity - relationship between volume and price movement
    volume_change = data['volume'].pct_change()
    factors['volume_elasticity'] = price_change / volume_change.where(volume_change != 0, np.nan)
    
    # 3. Multi-Timeframe Volatility
    # Range Convergence - compare short-term vs medium-term volatility
    short_range = (data['high'] - data['low']).rolling(window=5).mean()
    medium_range = (data['high'] - data['low']).rolling(window=20).mean()
    factors['range_convergence'] = short_range / medium_range
    
    # Volatility Compression - measure of volatility clustering
    daily_vol = (data['high'] - data['low']) / data['close']
    factors['vol_compression'] = daily_vol.rolling(window=10).std() / daily_vol.rolling(window=20).std()
    
    # 4. Overnight Reversal
    # Close-Open Extreme - measure of overnight gap extremes
    overnight_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    factors['close_open_extreme'] = overnight_gap.rolling(window=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else np.nan
    )
    
    # Reversal Intensity - strength of reversal after overnight moves
    day_return = (data['close'] - data['open']) / data['open']
    factors['reversal_intensity'] = -overnight_gap * day_return
    
    # Combine factors using equal weighting (can be optimized)
    final_factor = (
        factors['gap_range_ratio'].rank(pct=True) +
        factors['price_path_optimality'].rank(pct=True) +
        factors['amount_impact_dev'].rank(pct=True) +
        factors['volume_elasticity'].rank(pct=True) +
        factors['range_convergence'].rank(pct=True) +
        factors['vol_compression'].rank(pct=True) +
        factors['close_open_extreme'].rank(pct=True) +
        factors['reversal_intensity'].rank(pct=True)
    ) / 8
    
    return final_factor
