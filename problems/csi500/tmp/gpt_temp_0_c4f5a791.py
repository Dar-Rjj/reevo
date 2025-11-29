import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate cross-sectional alpha factors using price, volume, and amount data.
    Factors are designed to be novel, interpretable, and implementable.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Calculate daily returns and basic metrics
    data['returns'] = data['close'].pct_change()
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_close'] = data['close'].shift(1)
    
    # 1. Volatility-Adjusted Momentum
    # Calculate True Range
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Rolling volatility (20-day)
    data['volatility'] = data['true_range'].rolling(window=20, min_periods=10).mean()
    
    # Price momentum (5-day returns)
    data['momentum_5d'] = data['close'].pct_change(5)
    
    # Volatility-adjusted momentum
    data['vol_adj_momentum'] = data['momentum_5d'] / (data['volatility'] + 1e-8)
    
    # 2. Gap Reversal Factor
    data['opening_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['gap_reversal'] = data['opening_gap'] * (-data['intraday_return'])
    
    # 3. Volume-Weighted Price Range
    data['price_range'] = data['high'] - data['low']
    data['volume_range'] = data['price_range'] * data['volume']
    data['avg_volume_range'] = data['volume_range'].rolling(window=20, min_periods=10).mean()
    data['volume_weighted_range'] = data['volume_range'] / (data['avg_volume_range'] + 1e-8)
    
    # 4. Amount Efficiency Ratio
    data['amount_efficiency'] = data['price_range'] / (data['amount'] + 1e-8)
    
    # 5. Early-Late Session Divergence
    data['morning_return'] = (data['high'] - data['open']) / data['open']
    data['afternoon_return'] = (data['close'] - data['high']) / data['high']
    data['session_divergence'] = data['morning_return'] * (-data['afternoon_return'])
    
    # 6. Range Breakout Confirmation
    data['new_high'] = (data['high'] > data['prev_high']).astype(int)
    data['new_low'] = (data['low'] < data['prev_low']).astype(int)
    data['breakout_magnitude'] = data['new_high'] * (data['high'] - data['prev_high']) / data['prev_high'] - \
                                data['new_low'] * (data['prev_low'] - data['low']) / data['prev_low']
    
    # Volume spike (current volume vs 20-day average)
    data['avg_volume_20d'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_spike'] = data['volume'] / (data['avg_volume_20d'] + 1e-8)
    data['range_breakout'] = data['breakout_magnitude'] * data['volume_spike']
    
    # 7. Momentum-Volume Alignment
    data['momentum_signal'] = np.sign(data['momentum_5d'])
    data['volume_trend'] = np.sign(data['volume'] - data['volume'].shift(5))
    data['momentum_volume_alignment'] = data['momentum_signal'] * data['volume_trend']
    # Zero out when signals diverge
    data['momentum_volume_alignment'] = np.where(
        data['momentum_signal'] == data['volume_trend'], 
        data['momentum_volume_alignment'], 
        0
    )
    
    # 8. Range-Amplitude Efficiency
    data['range_normalized'] = data['price_range'] / (data['open'] + 1e-8)
    data['amount_per_move'] = data['amount'] / (data['price_range'] + 1e-8)
    data['range_amplitude_efficiency'] = data['range_normalized'] * data['amount_per_move']
    
    # Combine factors with equal weights
    factors = [
        'vol_adj_momentum',
        'gap_reversal', 
        'volume_weighted_range',
        'amount_efficiency',
        'session_divergence',
        'range_breakout',
        'momentum_volume_alignment',
        'range_amplitude_efficiency'
    ]
    
    # Standardize each factor and combine
    for date in data.index:
        day_data = data.loc[date]
        if len(day_data.shape) == 1:  # Single row
            combined_factor = 0
            for factor in factors:
                if pd.notna(day_data[factor]):
                    combined_factor += day_data[factor]
            factor_values[date] = combined_factor / len(factors)
        else:  # Multiple rows (cross-sectional)
            day_factors = day_data[factors].copy()
            # Remove any infinite values
            day_factors = day_factors.replace([np.inf, -np.inf], np.nan)
            # Z-score normalize each factor cross-sectionally
            for factor in factors:
                if factor in day_factors.columns:
                    valid_data = day_factors[factor].dropna()
                    if len(valid_data) > 1:
                        mean_val = valid_data.mean()
                        std_val = valid_data.std()
                        if std_val > 0:
                            day_factors[factor] = (day_factors[factor] - mean_val) / std_val
                        else:
                            day_factors[factor] = 0
                    else:
                        day_factors[factor] = 0
            
            # Equal-weighted combination
            combined_factors = day_factors[factors].mean(axis=1, skipna=True)
            factor_values[date] = combined_factors.mean() if not combined_factors.empty else 0
    
    return factor_values
