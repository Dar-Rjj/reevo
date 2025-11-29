import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Adjusted Opening Momentum Divergence
    # Opening Momentum Strength
    data['close_shift'] = data['close'].shift(1)
    data['opening_momentum'] = (data['open'] - data['close_shift']) / data['close_shift']
    
    # True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close_shift'])
    data['tr3'] = abs(data['low'] - data['close_shift'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility Regime
    data['atr_10'] = data['true_range'].rolling(window=10).mean()
    data['vol_ratio'] = data['true_range'] / data['atr_10']
    
    # Volume change sign
    data['volume_shift'] = data['volume'].shift(1)
    data['volume_change_sign'] = np.sign(data['volume'] / data['volume_shift'] - 1)
    
    # Combined signal
    data['factor1'] = np.arcsinh(data['opening_momentum'] * data['vol_ratio']) * data['volume_change_sign']
    
    # Amount-Weighted Price Reversal Oscillator
    # Price Reversal Intensity
    data['close_shift2'] = data['close'].shift(2)
    data['reversal_intensity'] = (data['close'] - data['close_shift']) * (data['close_shift'] - data['close_shift2'])
    
    # Amount-weighted Reversal
    data['amount_avg_10'] = data['amount'].rolling(window=10).mean()
    data['weighted_reversal'] = (data['reversal_intensity'] * data['amount']) / data['amount_avg_10']
    
    # Oscillator Signal
    data['ema_5'] = data['weighted_reversal'].ewm(span=5).mean()
    data['deviation'] = (data['weighted_reversal'] - data['ema_5']) / abs(data['ema_5'])
    data['factor2'] = 1 / (1 + np.exp(-data['deviation']))
    
    # High-Low Compression Breakout Detector
    # Price Range Compression
    data['high_5d_max'] = data['high'].rolling(window=5).max()
    data['low_5d_min'] = data['low'].rolling(window=5).min()
    data['compression_ratio'] = (data['high'] - data['low']) / (data['high_5d_max'] - data['low_5d_min'])
    
    # Volume Confirmation
    data['volume_rank'] = data['volume'].rolling(window=15).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    )
    
    # Breakout Signal
    data['price_direction'] = np.sign(data['close'] - data['close_shift'])
    data['factor3'] = np.sqrt(data['compression_ratio'] * data['volume_rank']) * data['price_direction']
    
    # Multi-Scale Volatility Regime Factor
    # Short-term Volatility (3-day)
    data['daily_range_pct'] = (data['high'] - data['low']) / data['close']
    data['vol_short'] = data['daily_range_pct'].rolling(window=3).mean()
    
    # Medium-term Volatility (10-day)
    data['vol_medium'] = data['daily_range_pct'].rolling(window=10).mean()
    
    # Volatility Regime Shift
    data['vol_regime_ratio'] = data['vol_short'] / data['vol_medium']
    data['vol_acceleration'] = (data['vol_short'] - data['vol_medium']) / data['vol_medium']
    
    # Volume trend
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['volume_trend'] = data['volume'] / data['volume_5d_avg']
    
    # Combined signal
    data['factor4'] = np.cbrt(data['vol_regime_ratio'] * data['vol_acceleration']) * data['volume_trend']
    
    # Opening-Closing Divergence Momentum
    # Intraday Momentum
    data['intraday_momentum'] = (data['close'] - data['open']) / data['open']
    
    # Overnight Momentum
    data['overnight_momentum'] = (data['open'] - data['close_shift']) / data['close_shift']
    
    # Momentum Divergence
    data['momentum_diff'] = data['intraday_momentum'] - data['overnight_momentum']
    data['momentum_ratio'] = data['intraday_momentum'] / data['overnight_momentum']
    
    # Amount intensity
    data['amount_10d_avg'] = data['amount'].rolling(window=10).mean()
    data['amount_intensity'] = data['amount'] / data['amount_10d_avg']
    
    # Combined signal
    data['factor5'] = np.arctan(data['momentum_diff'] * data['momentum_ratio']) * data['amount_intensity']
    
    # Combine all factors with equal weights
    factors = ['factor1', 'factor2', 'factor3', 'factor4', 'factor5']
    data['combined_factor'] = data[factors].mean(axis=1)
    
    return data['combined_factor']
