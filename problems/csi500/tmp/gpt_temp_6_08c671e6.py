import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns
    data['returns'] = data['close'].pct_change()
    data['prev_close'] = data['close'].shift(1)
    
    # Calculate average price (using OHLC average)
    data['avg_price'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    
    # Volume per price unit (liquidity measure)
    data['volume_per_price'] = data['volume'] / data['avg_price']
    
    # Initialize factor series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    for date in data.index:
        current_data = data.loc[:date].copy()
        
        if len(current_data) < 3:  # Need at least 3 days for meaningful calculations
            factor_values.loc[date] = 0
            continue
            
        # Factor 1: Volume-Weighted Volatility Asymmetry
        current_data['is_up_day'] = current_data['close'] > current_data['prev_close']
        current_data['is_down_day'] = current_data['close'] < current_data['prev_close']
        current_data['abs_return'] = abs(current_data['returns'])
        
        up_vol = current_data.loc[current_data['is_up_day'], 'abs_return'].sum()
        down_vol = current_data.loc[current_data['is_down_day'], 'abs_return'].sum()
        volatility_asymmetry = up_vol - down_vol
        
        # Use recent volume per price for weighting
        recent_vpp = current_data['volume_per_price'].iloc[-1] if not current_data.empty else 1
        factor1 = volatility_asymmetry * recent_vpp
        
        # Factor 2: Volume-Weighted Opening Resolution
        current_data['opening_gap'] = current_data['open'] - current_data['prev_close']
        current_data['gap_magnitude'] = abs(current_data['opening_gap'])
        
        resolution_efficiency = 0
        if len(current_data) >= 2:
            latest = current_data.iloc[-1]
            if latest['opening_gap'] > 0 and latest['gap_magnitude'] > 0:
                # Positive gap: measure upward resolution
                upward_extension = latest['high'] - latest['open']
                resolution_efficiency = upward_extension / latest['gap_magnitude']
            elif latest['opening_gap'] < 0 and latest['gap_magnitude'] > 0:
                # Negative gap: measure downward resolution
                downward_extension = latest['open'] - latest['low']
                resolution_efficiency = downward_extension / latest['gap_magnitude']
        
        factor2 = resolution_efficiency * recent_vpp
        
        # Factor 3: Volume-Price Momentum Persistence
        current_data['return_direction'] = np.sign(current_data['returns'])
        current_data['direction_change'] = current_data['return_direction'] != current_data['return_direction'].shift(1)
        current_data['streak_id'] = current_data['direction_change'].cumsum()
        
        # Calculate streak statistics
        streak_stats = current_data.groupby('streak_id').agg({
            'returns': ['count', lambda x: abs(x).mean()]
        }).reset_index()
        streak_stats.columns = ['streak_id', 'streak_length', 'avg_abs_return']
        
        if not streak_stats.empty:
            current_streak_id = current_data['streak_id'].iloc[-1]
            current_streak = streak_stats[streak_stats['streak_id'] == current_streak_id]
            if not current_streak.empty:
                persistence = current_streak['streak_length'].iloc[0] * current_streak['avg_abs_return'].iloc[0]
            else:
                persistence = 0
        else:
            persistence = 0
            
        factor3 = persistence * recent_vpp
        
        # Factor 4: Volume-Weighted Fractal Efficiency
        current_data['price_efficiency'] = abs(current_data['close'] - current_data['prev_close']) / (current_data['high'] - current_data['low'])
        current_data['price_efficiency'] = current_data['price_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Volume fractal (volume relative to average)
        avg_volume = current_data['volume'].mean()
        volume_fractal = current_data['volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1
        
        latest_efficiency = current_data['price_efficiency'].iloc[-1] if not current_data.empty else 0
        factor4 = latest_efficiency * volume_fractal
        
        # Combine factors (equal weighting)
        combined_factor = (factor1 + factor2 + factor3 + factor4) / 4
        factor_values.loc[date] = combined_factor
    
    return factor_values
