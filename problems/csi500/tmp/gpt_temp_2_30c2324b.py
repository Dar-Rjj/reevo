import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Helper function to count consecutive signs
    def count_consecutive(series, window):
        signs = np.sign(series)
        result = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if i < window - 1:
                result.iloc[i] = 0
            else:
                window_data = signs.iloc[i-window+1:i+1]
                if len(window_data) == window and not window_data.isna().any():
                    if all(x > 0 for x in window_data):
                        result.iloc[i] = window
                    elif all(x < 0 for x in window_data):
                        result.iloc[i] = -window
                    else:
                        result.iloc[i] = 0
                else:
                    result.iloc[i] = 0
        return result
    
    # Intraday Price Efficiency & Volume Confirmation
    high_low_range = data['high'] - data['low']
    high_low_range = high_low_range.replace(0, np.nan)
    
    # Price Range Efficiency Calculation
    intraday_price_capture = (data['close'] - data['low']) / high_low_range
    opening_gap_efficiency = (data['open'] - data['low']) / high_low_range
    closing_efficiency = (data['close'] - data['open']) / high_low_range
    
    # Volume-Weighted Efficiency Metrics
    volume_weighted_price_capture = intraday_price_capture * data['volume']
    
    amount_volume_ratio = data['amount'] / data['volume']
    amount_volume_ratio = amount_volume_ratio.replace(0, np.nan)
    volume_adjusted_range_efficiency = (data['high'] - data['low']) * data['volume'] / amount_volume_ratio
    
    volume_rolling_5 = data['volume'].rolling(5, min_periods=1).mean()
    volume_concentration_ratio = data['volume'] / volume_rolling_5
    
    # Efficiency-Volume Divergence Detection
    efficiency_volume_corr = np.sign(closing_efficiency) * np.sign(volume_concentration_ratio - 1)
    
    closing_efficiency_std = closing_efficiency.rolling(5, min_periods=1).std()
    closing_efficiency_std = closing_efficiency_std.replace(0, np.nan)
    abnormal_efficiency = np.abs(closing_efficiency) / closing_efficiency_std
    
    volume_efficiency_momentum = volume_concentration_ratio * closing_efficiency
    
    # Momentum Regime Identification & Switching
    short_term_momentum = data['close'] / data['close'].shift(2) - 1
    medium_term_momentum = data['close'] / data['close'].shift(5) - 1
    
    momentum_regime_score = np.sign(short_term_momentum) * np.sign(medium_term_momentum)
    
    price_change_sign = np.sign(data['close'] - data['close'].shift(1))
    momentum_persistence = count_consecutive(price_change_sign, 3)
    
    # Regime-Dependent Efficiency Patterns
    bull_regime = (closing_efficiency > opening_gap_efficiency).astype(int)
    bear_regime = (closing_efficiency < opening_gap_efficiency).astype(int)
    transition_regime = (np.abs(closing_efficiency - opening_gap_efficiency) < 0.1).astype(int)
    
    # Regime-Switching Signals
    bullish_efficiency_signal = volume_weighted_price_capture * momentum_persistence
    bearish_efficiency_signal = -volume_weighted_price_capture * momentum_persistence
    transition_signal = volume_efficiency_momentum * momentum_regime_score
    
    # Price-Volume Acceleration & Deceleration
    volume_momentum = data['volume'] / data['volume'].shift(1) - 1
    volume_acceleration = volume_momentum - volume_momentum.shift(1)
    
    volume_momentum_sign = np.sign(volume_momentum)
    volume_persistence = count_consecutive(volume_momentum_sign, 3)
    
    intraday_price_acceleration = (data['close'] - data['open']) / data['open'] - (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    range_acceleration = (data['high'] - data['low']) / data['close'].shift(1) - (data['high'].shift(1) - data['low'].shift(1)) / data['close'].shift(2)
    
    acceleration_divergence = intraday_price_acceleration - range_acceleration
    
    volume_price_acceleration_corr = np.sign(volume_acceleration) * np.sign(intraday_price_acceleration)
    acceleration_momentum_score = volume_persistence * acceleration_divergence
    deceleration_warning = -volume_acceleration * np.abs(intraday_price_acceleration)
    
    # Efficiency-Based Support/Resistance Framework
    efficiency_based_resistance = data['high'].rolling(3, min_periods=1).max() * (1 + volume_concentration_ratio)
    efficiency_based_support = data['low'].rolling(3, min_periods=1).min() * (1 - volume_concentration_ratio)
    
    resistance_support_range = efficiency_based_resistance - efficiency_based_support
    resistance_support_range = resistance_support_range.replace(0, np.nan)
    efficiency_zone = (data['close'] - efficiency_based_support) / resistance_support_range
    
    upper_efficiency_breakout = (data['close'] > efficiency_based_resistance).astype(int)
    lower_efficiency_breakout = (data['close'] < efficiency_based_support).astype(int)
    
    efficiency_breakout_strength = np.abs(data['close'] - efficiency_based_support) / np.abs(resistance_support_range)
    
    volume_confirmed_breakout = efficiency_breakout_strength * volume_concentration_ratio
    false_breakout_detection = efficiency_breakout_strength * (1 - volume_concentration_ratio)
    
    breakout_combined = upper_efficiency_breakout | lower_efficiency_breakout
    breakout_persistence = count_consecutive(breakout_combined, 2)
    
    # Cross-Timeframe Efficiency Convergence
    efficiency_convergence_2day = closing_efficiency * closing_efficiency.shift(1)
    efficiency_trend_3day = closing_efficiency.rolling(3, min_periods=1).mean()
    efficiency_momentum = closing_efficiency - closing_efficiency.shift(2)
    
    short_term_volume_efficiency = volume_concentration_ratio * closing_efficiency
    
    volume_ratio_5_10 = data['volume'].rolling(5, min_periods=1).mean() / data['volume'].rolling(10, min_periods=1).mean()
    medium_term_volume_efficiency = volume_ratio_5_10 * efficiency_momentum
    
    volume_efficiency_convergence = np.sign(short_term_volume_efficiency) * np.sign(medium_term_volume_efficiency)
    
    # Regime-Adaptive Convergence Signals
    bullish_convergence = volume_efficiency_convergence * efficiency_momentum * momentum_persistence
    bearish_convergence = -volume_efficiency_convergence * efficiency_momentum * momentum_persistence
    neutral_convergence = medium_term_volume_efficiency * acceleration_momentum_score
    
    # Adaptive Alpha Synthesis
    # Core Alpha Construction
    efficiency_foundation = volume_weighted_price_capture * closing_efficiency
    momentum_enhancement = efficiency_foundation * momentum_regime_score * volume_efficiency_convergence
    breakout_integration = momentum_enhancement * efficiency_breakout_strength * breakout_persistence
    acceleration_finalization = breakout_integration * acceleration_momentum_score * volume_price_acceleration_corr
    
    # Regime-Based Component Selection
    bull_alpha = (0.4 * bullish_efficiency_signal + 
                  0.3 * volume_confirmed_breakout + 
                  0.2 * bullish_convergence + 
                  0.1 * acceleration_momentum_score)
    
    bear_alpha = (0.4 * bearish_efficiency_signal + 
                  0.3 * volume_confirmed_breakout + 
                  0.2 * bearish_convergence + 
                  0.1 * acceleration_momentum_score)
    
    transition_alpha = (0.3 * transition_signal + 
                        0.25 * neutral_convergence + 
                        0.2 * efficiency_zone + 
                        0.15 * volume_efficiency_momentum + 
                        0.1 * deceleration_warning)
    
    # Final Factor Output with regime-specific scaling
    final_alpha = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if bull_regime.iloc[i] == 1:
            final_alpha.iloc[i] = bull_alpha.iloc[i]
        elif bear_regime.iloc[i] == 1:
            final_alpha.iloc[i] = bear_alpha.iloc[i]
        else:
            final_alpha.iloc[i] = transition_alpha.iloc[i]
    
    # Normalize the final alpha
    final_alpha = (final_alpha - final_alpha.mean()) / final_alpha.std()
    
    return final_alpha
