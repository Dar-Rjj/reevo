import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Intraday Trend Persistence Factor
    # Calculate normalized price position
    epsilon = 1e-8
    normalized_position = (data['close'] - data['low']) / (data['high'] - data['low'] + epsilon)
    
    # Calculate rolling autocorrelation for trend persistence
    window_size = 5
    autocorr_values = pd.Series(index=data.index, dtype=float)
    
    for i in range(window_size, len(data)):
        window_data = normalized_position.iloc[i-window_size:i]
        if len(window_data) >= 2:
            autocorr = window_data.autocorr(lag=1)
            autocorr_values.iloc[i] = autocorr if not np.isnan(autocorr) else 0
    
    # Trend magnitude proxy
    trend_magnitude = (data['high'] - data['low']) / data['close']
    
    # Volume confirmation
    volume_avg = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_ratio = data['volume'] / (volume_avg + epsilon)
    
    # Combine for intraday trend persistence factor
    trend_persistence = autocorr_values * trend_magnitude * volume_ratio
    
    # Volatility Regime Momentum
    # Calculate returns
    returns = data['close'].pct_change()
    
    # Calculate volatility measures
    short_vol = returns.abs().rolling(window=10, min_periods=1).std()
    long_vol = returns.abs().rolling(window=60, min_periods=1).std()
    vol_ratio = short_vol / (long_vol + epsilon)
    
    # Calculate momentum measures
    ret_3d = data['close'].pct_change(3)
    ret_5d = data['close'].pct_change(5)
    
    # High volatility regime momentum
    high_vol_mask = vol_ratio > 1.2
    high_vol_momentum = ret_3d * vol_ratio
    high_vol_momentum = high_vol_momentum.where(high_vol_mask, 0)
    
    # Low volatility regime momentum
    low_vol_mask = vol_ratio < 0.8
    low_vol_momentum = -ret_5d * (1 - vol_ratio)  # Mean reversion adjustment
    low_vol_momentum = low_vol_momentum.where(low_vol_mask, 0)
    
    # Combine regime signals
    regime_momentum = high_vol_momentum + low_vol_momentum
    
    # Price-Volume Divergence Oscillator
    # Price momentum
    price_momentum_3d = data['close'].pct_change(3)
    price_momentum_5d = data['close'].pct_change(5)
    
    # Volume momentum
    volume_momentum_3d = data['volume'].pct_change(3)
    volume_momentum_5d = data['volume'].pct_change(5)
    
    # Detect divergence
    bullish_divergence = ((price_momentum_3d < 0) & (volume_momentum_3d > 0)) | \
                        ((price_momentum_5d < 0) & (volume_momentum_5d > 0))
    
    bearish_divergence = ((price_momentum_3d > 0) & (volume_momentum_3d < 0)) | \
                        ((price_momentum_5d > 0) & (volume_momentum_5d < 0))
    
    # Calculate divergence magnitude
    divergence_strength = pd.Series(0, index=data.index)
    divergence_strength = divergence_strength.where(~bullish_divergence, 
                                                   (volume_momentum_3d.abs() + volume_momentum_5d.abs()) / 2)
    divergence_strength = divergence_strength.where(~bearish_divergence, 
                                                   -(volume_momentum_3d.abs() + volume_momentum_5d.abs()) / 2)
    
    # Smooth oscillator
    divergence_oscillator = divergence_strength.rolling(window=3, min_periods=1).mean()
    
    # Amplitude-Adjusted Reversal Factor
    # Measure price swings
    daily_range = (data['high'] - data['low']) / data['close']
    avg_range = daily_range.rolling(window=5, min_periods=1).mean()
    
    # Identify extreme moves
    extreme_moves = daily_range > (1.5 * avg_range)
    move_magnitude = daily_range / (avg_range + epsilon)
    
    # Oversold/overbought conditions
    ma_5 = data['close'].rolling(window=5, min_periods=1).mean()
    ma_10 = data['close'].rolling(window=10, min_periods=1).mean()
    
    position_from_ma = (data['close'] - ma_5) / (ma_10 - ma_5 + epsilon)
    
    # Volume pattern analysis
    volume_ma_5 = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_ratio_extreme = data['volume'] / (volume_ma_5 + epsilon)
    
    # Generate reversal signal
    oversold = (position_from_ma < -0.1) & extreme_moves
    overbought = (position_from_ma > 0.1) & extreme_moves
    
    reversal_signal = pd.Series(0, index=data.index)
    reversal_signal = reversal_signal.where(~oversold, move_magnitude * volume_ratio_extreme)
    reversal_signal = reversal_signal.where(~overbought, -move_magnitude * volume_ratio_extreme)
    
    # Combine all factors with equal weights
    factor = (trend_persistence.fillna(0) + 
              regime_momentum.fillna(0) + 
              divergence_oscillator.fillna(0) + 
              reversal_signal.fillna(0)) / 4
    
    return factor
