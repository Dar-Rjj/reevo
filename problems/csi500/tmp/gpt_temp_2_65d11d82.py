import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum Reversal with Liquidity Acceleration factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize output series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Calculate required intermediate variables
    data['prev_close'] = data['close'].shift(1)
    data['price_range'] = data['high'] - data['low']
    data['intraday_move'] = data['close'] - data['open']
    
    # 1. Intraday Momentum Reversal Patterns
    # Price Range Efficiency
    data['range_efficiency'] = np.where(
        data['price_range'] > 0,
        data['intraday_move'] / data['price_range'],
        0
    )
    
    # Opening Gap Analysis
    data['gap_size'] = np.where(
        data['prev_close'] > 0,
        data['open'] / data['prev_close'] - 1,
        0
    )
    
    # Gap Absorption
    data['gap_absorption'] = np.where(
        (data['gap_size'] > 0) & (data['price_range'] > 0),
        (data['close'] - data['low']) / (data['high'] - data['low']),
        np.where(
            (data['gap_size'] < 0) & (data['price_range'] > 0),
            (data['high'] - data['close']) / (data['high'] - data['low']),
            0.5
        )
    )
    
    # Morning vs Afternoon Performance
    data['morning_strength'] = np.where(
        data['price_range'] > 0,
        (data['high'] - data['open']) / data['price_range'],
        0
    )
    
    data['afternoon_recovery'] = np.where(
        data['price_range'] > 0,
        (data['close'] - data['low']) / data['price_range'],
        0
    )
    
    # Failed Breakouts
    data['high_rejection'] = np.where(
        data['price_range'] > 0,
        (data['high'] - data['close']) / data['price_range'],
        0
    )
    
    # Momentum Reversal Score
    data['reversal_prob'] = (
        (1 - data['range_efficiency'].abs()) * 0.3 +
        data['gap_absorption'] * 0.25 +
        (data['afternoon_recovery'] - data['morning_strength']) * 0.25 +
        data['high_rejection'] * 0.2
    )
    
    # Directional Momentum Reversal Factor
    data['momentum_reversal'] = np.where(
        data['intraday_move'] > 0,
        -data['reversal_prob'],  # Negative for overbought reversal
        data['reversal_prob']    # Positive for oversold reversal
    )
    
    # 2. Liquidity Acceleration Dynamics
    # Volume Profile Analysis
    data['trade_size'] = np.where(
        data['volume'] > 0,
        data['amount'] / data['volume'],
        0
    )
    
    # Rolling average trade size (5-day)
    data['avg_trade_size'] = data['trade_size'].rolling(window=5, min_periods=3).mean()
    data['trade_size_deviation'] = data['trade_size'] / data['avg_trade_size'] - 1
    
    # Volume-Price Efficiency
    # Simple VWAP approximation using (high + low + close)/3
    data['vwap_approx'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap_deviation'] = np.where(
        data['price_range'] > 0,
        (data['close'] - data['vwap_approx']) / data['price_range'],
        0
    )
    
    # Price move per unit volume
    data['price_per_volume'] = np.where(
        data['volume'] > 0,
        data['intraday_move'].abs() / data['volume'],
        0
    )
    
    # Liquidity Efficiency Score
    data['liquidity_efficiency'] = (
        -data['vwap_deviation'].abs() * 0.4 +
        -data['trade_size_deviation'].abs() * 0.3 +
        -data['price_per_volume'] * 0.3
    )
    
    # Volume pattern analysis (5-day rolling)
    data['volume_ma'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_deviation'] = data['volume'] / data['volume_ma'] - 1
    
    # Liquidity Acceleration Signal
    data['liquidity_acceleration'] = (
        data['volume_deviation'] * 0.4 +
        data['liquidity_efficiency'] * 0.6
    )
    
    # 3. Signal Synthesis
    # Recent performance tracking (3-day correlation approximation)
    for i in range(len(data)):
        if i >= 3:
            recent_data = data.iloc[i-3:i]
            if len(recent_data) >= 2:
                # Simple momentum effectiveness measure
                mom_perf = recent_data['momentum_reversal'].mean()
                liq_perf = recent_data['liquidity_acceleration'].mean()
                
                # Dynamic weighting based on recent performance
                mom_weight = 0.5 + (mom_perf * 0.2)
                liq_weight = 0.5 + (liq_perf * 0.2)
                
                # Normalize weights
                total_weight = abs(mom_weight) + abs(liq_weight)
                if total_weight > 0:
                    mom_weight = mom_weight / total_weight
                    liq_weight = liq_weight / total_weight
                else:
                    mom_weight, liq_weight = 0.5, 0.5
            else:
                mom_weight, liq_weight = 0.5, 0.5
        else:
            mom_weight, liq_weight = 0.5, 0.5
        
        # Calculate composite signal
        if i >= 1:  # Ensure we have at least one previous value
            mom_signal = data['momentum_reversal'].iloc[i]
            liq_signal = data['liquidity_acceleration'].iloc[i]
            
            # Signal alignment check
            signal_alignment = 1 if (mom_signal * liq_signal > 0) else 0.7
            
            # Composite factor with dynamic weighting
            composite = (
                mom_signal * mom_weight * signal_alignment +
                liq_signal * liq_weight * signal_alignment
            )
            
            # Non-linear transformation for extreme values
            if abs(composite) > 2:
                composite = np.sign(composite) * (2 + np.log1p(abs(composite) - 2))
            
            alpha.iloc[i] = composite
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    # Ensure stationarity through simple differencing
    if len(alpha) > 1:
        alpha = alpha - alpha.shift(1).fillna(0)
    
    return alpha
