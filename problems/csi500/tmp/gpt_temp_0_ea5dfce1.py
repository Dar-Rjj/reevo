import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns
    returns = data['close'].pct_change()
    
    # Volatility Regime Classification
    short_term_vol = returns.rolling(5).std()
    medium_term_vol = returns.rolling(20).std()
    volatility_ratio = short_term_vol / medium_term_vol
    
    # Breakout Strength Components
    morning_breakout_momentum = (data['high'] - data['open']) / data['open']
    afternoon_support_momentum = (data['close'] - data['low']) / data['close']
    
    # Historical Breakout Context
    rolling_5d_high = data['high'].rolling(5).max()
    rolling_5d_low = data['low'].rolling(5).min()
    current_high_breakout_ratio = (data['high'] - rolling_5d_high) / (rolling_5d_high - rolling_5d_low)
    current_low_breakout_ratio = (data['low'] - rolling_5d_low) / (rolling_5d_high - rolling_5d_low)
    breakout_ratio = (current_high_breakout_ratio + current_low_breakout_ratio) / 2
    
    # Volatility Persistence Analysis
    true_range = np.maximum(data['high'] - data['low'], 
                           np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                     abs(data['low'] - data['close'].shift(1))))
    atr = true_range.rolling(5).mean()
    volatility_persistence = returns.rolling(5).std() / returns.rolling(10).std()
    
    # Range Efficiency and Momentum Synthesis
    price_range_efficiency = (data['close'] - data['low']) / (data['high'] - data['low'])
    range_efficiency_momentum = price_range_efficiency - price_range_efficiency.shift(1)
    volatility_adjusted_range_efficiency = price_range_efficiency / atr
    
    # Momentum Alignment Components
    short_term_price_momentum = data['close'] / data['close'].shift(3) - 1
    today_range = data['high'] - data['low']
    range_3d_ago = (data['high'].shift(3) - data['low'].shift(3))
    range_momentum = (today_range - range_3d_ago) / range_3d_ago
    momentum_alignment = np.sign(short_term_price_momentum) * np.sign(range_momentum)
    
    # Reversal-Momentum Integration
    intraday_return = (data['close'] - data['open']) / data['open']
    previous_day_momentum = (data['close'].shift(1) - data['open'].shift(1)) / data['open'].shift(1)
    reversal_indicator = -np.sign(previous_day_momentum) * intraday_return
    
    # Volume and Liquidity Confirmation
    volume_acceleration = data['volume'] / data['volume'].shift(1) - 1
    rolling_3d_volume_growth = data['volume'].pct_change(3)
    large_trade_concentration = data['amount'].rolling(5).sum() / data['volume'].rolling(5).sum()
    
    # Volume Breakout Analysis
    volume_percentile = data['volume'].rolling(20).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()))
    high_volume_periods = data['volume'] > 1.5 * data['volume'].rolling(5).mean()
    volume_weighted_range_efficiency = price_range_efficiency * data['volume'][high_volume_periods].mean()
    
    # Liquidity-Scaled Components
    volume_weighted_price_range = (data['high'] - data['low']) * data['volume']
    amount_efficiency = data['amount'] / (data['high'] - data['low'])
    liquidity_scaled_signal = reversal_indicator * amount_efficiency / volume_weighted_price_range
    
    # Regime-Adaptive Component Integration
    high_vol_mode = volatility_ratio > 1.2
    low_vol_mode = volatility_ratio < 0.8
    normal_vol_mode = (volatility_ratio >= 0.8) & (volatility_ratio <= 1.2)
    
    # Initialize factor components
    primary_component = pd.Series(index=data.index, dtype=float)
    secondary_component = pd.Series(index=data.index, dtype=float)
    regime_factor = pd.Series(index=data.index, dtype=float)
    
    # High Volatility Mode
    primary_component[high_vol_mode] = (volatility_adjusted_range_efficiency * volume_acceleration)[high_vol_mode]
    secondary_component[high_vol_mode] = (range_efficiency_momentum * momentum_alignment)[high_vol_mode]
    regime_factor[high_vol_mode] = (primary_component * secondary_component * volatility_persistence)[high_vol_mode]
    
    # Low Volatility Mode
    primary_component[low_vol_mode] = (volatility_adjusted_range_efficiency * large_trade_concentration)[low_vol_mode]
    secondary_component[low_vol_mode] = (volume_weighted_range_efficiency * morning_breakout_momentum)[low_vol_mode]
    regime_factor[low_vol_mode] = (primary_component * secondary_component * breakout_ratio)[low_vol_mode]
    
    # Normal Volatility Mode
    primary_component[normal_vol_mode] = (volatility_adjusted_range_efficiency * (volume_acceleration + large_trade_concentration))[normal_vol_mode]
    secondary_component[normal_vol_mode] = (price_range_efficiency * afternoon_support_momentum)[normal_vol_mode]
    regime_factor[normal_vol_mode] = (primary_component * secondary_component * momentum_alignment)[normal_vol_mode]
    
    # Final Alpha Construction
    core_range_breakout_component = regime_factor * reversal_indicator
    volume_confirmation_enhancement = core_range_breakout_component * liquidity_scaled_signal
    
    # Breakout Persistence Integration
    breakout_persistence = current_high_breakout_ratio.rolling(10).apply(lambda x: (x > 0).sum() / 10)
    
    # Volume-Momentum Confirmation Score
    volume_momentum_confirmation = np.where(
        np.sign(volume_acceleration) == np.sign(short_term_price_momentum), 1, -1
    )
    
    # Final Alpha
    final_alpha = volume_confirmation_enhancement * breakout_persistence * volume_momentum_confirmation
    
    return final_alpha
