import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price features
    data['prev_close'] = data['close'].shift(1)
    data['price_range'] = data['high'] - data['low']
    data['intraday_move'] = data['close'] - data['open']
    
    # 1. Overnight Gap and Intraday Fade Patterns
    # Overnight Gap Strength
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_magnitude'] = abs(data['overnight_gap'])
    
    # Intraday Fade Completion
    data['fade_completion'] = np.where(
        data['price_range'] != 0,
        data['intraday_move'] / data['price_range'],
        0
    )
    
    # 2. Volume Divergence During Price Moves
    # Price-Volume Directional Alignment (rolling correlation)
    data['price_change'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    
    # Calculate rolling correlation between price and volume changes
    window = 15
    price_volume_corr = []
    for i in range(len(data)):
        if i < window:
            price_volume_corr.append(0)
        else:
            start_idx = i - window + 1
            end_idx = i + 1
            price_changes = data['price_change'].iloc[start_idx:end_idx]
            volume_changes = data['volume_change'].iloc[start_idx:end_idx]
            valid_mask = (~price_changes.isna()) & (~volume_changes.isna())
            if valid_mask.sum() > 5:
                corr = np.corrcoef(price_changes[valid_mask], volume_changes[valid_mask])[0, 1]
                price_volume_corr.append(corr if not np.isnan(corr) else 0)
            else:
                price_volume_corr.append(0)
    data['price_volume_alignment'] = price_volume_corr
    
    # Volume Concentration in Key Price Levels
    # Calculate volume distribution across price quartiles
    data['price_quartile'] = pd.qcut(data['close'], 4, labels=False, duplicates='drop')
    volume_by_quartile = data.groupby('price_quartile')['volume'].transform('mean')
    data['volume_concentration'] = data['volume'] / volume_by_quartile
    
    # 3. Liquidity Absorption at Support/Resistance
    # Volume-to-range ratio for absorption detection
    data['volume_range_ratio'] = np.where(
        data['price_range'] != 0,
        data['volume'] / data['price_range'],
        data['volume']
    )
    
    # Detect absorption zones (high volume with small price movement)
    data['absorption_strength'] = data['volume_range_ratio'] / data['volume_range_ratio'].rolling(window=20, min_periods=1).mean()
    
    # Support/Resistance Break Attempts
    # Calculate rolling high/low for resistance/support levels
    data['rolling_high_20'] = data['high'].rolling(window=20, min_periods=1).max()
    data['rolling_low_20'] = data['low'].rolling(window=20, min_periods=1).min()
    
    # Failed breakout detection
    data['near_resistance'] = (data['high'] >= data['rolling_high_20'] * 0.99) & (data['high'] <= data['rolling_high_20'] * 1.01)
    data['near_support'] = (data['low'] >= data['rolling_low_20'] * 0.99) & (data['low'] <= data['rolling_low_20'] * 1.01)
    
    data['failed_breakout'] = np.where(
        data['near_resistance'] & (data['close'] < data['open']),
        abs(data['intraday_move']) / data['price_range'].replace(0, 1),
        0
    )
    
    data['failed_breakdown'] = np.where(
        data['near_support'] & (data['close'] > data['open']),
        abs(data['intraday_move']) / data['price_range'].replace(0, 1),
        0
    )
    
    data['rejection_strength'] = data['failed_breakout'] + data['failed_breakdown']
    
    # 4. Generate Momentum-Fade Composite Signal
    # Combine Gap and Fade Components (contrarian signal)
    gap_fade_component = -data['overnight_gap'] * data['fade_completion']
    
    # Weight by Volume Divergence Evidence
    volume_weight = (1 + data['price_volume_alignment']) * data['volume_concentration']
    weighted_signal = gap_fade_component * volume_weight
    
    # Enhance with Liquidity Absorption Signals
    absorption_enhancement = data['absorption_strength'] * (1 + data['rejection_strength'])
    final_signal = weighted_signal * absorption_enhancement
    
    # Clean and return the factor
    factor = final_signal.replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor
