import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate basic components
    df = df.copy()
    df['daily_range'] = df['high'] - df['low']
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['price_position'] = df['price_position'].replace([np.inf, -np.inf], np.nan).fillna(0.5)
    
    # Range persistence calculation
    df['range_percentile'] = df['daily_range'].rolling(window=5, min_periods=3).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    )
    
    # Momentum reversal signal
    df['price_change'] = df['close'].pct_change()
    df['trend_3d'] = df['close'].pct_change(3)
    
    # Identify contrarian moves (price moves against 3-day trend)
    df['contrarian_signal'] = 0
    mask_contrarian = ((df['price_change'] > 0) & (df['trend_3d'] < 0)) | ((df['price_change'] < 0) & (df['trend_3d'] > 0))
    df.loc[mask_contrarian, 'contrarian_signal'] = np.abs(df['price_change'])
    
    # Combine with extreme price positions
    df['position_strength'] = 0
    extreme_low = df['price_position'] < 0.2
    extreme_high = df['price_position'] > 0.8
    neutral = (df['price_position'] >= 0.4) & (df['price_position'] <= 0.6)
    
    df.loc[extreme_low | extreme_high, 'position_strength'] = 1.5
    df.loc[neutral, 'position_strength'] = 0.5
    df.loc[~((extreme_low | extreme_high) | neutral), 'position_strength'] = 1.0
    
    df['momentum_reversal'] = df['contrarian_signal'] * df['position_strength']
    
    # Volume-weighted confirmation
    df['volume_avg_20d'] = df['volume'].rolling(window=20, min_periods=10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_avg_20d']
    df['volume_deviation'] = np.where(df['volume_ratio'] > 1.5, df['volume_ratio'], 1.0)
    
    # Directional consistency weighting
    df['range_change'] = df['daily_range'].pct_change()
    df['range_expanding'] = (df['range_change'] > 0).astype(int)
    df['range_contracting'] = (df['range_change'] < 0).astype(int)
    
    # Calculate consecutive streaks
    df['expanding_streak'] = 0
    df['contracting_streak'] = 0
    
    current_expand = 0
    current_contract = 0
    
    for i in range(len(df)):
        if df['range_expanding'].iloc[i] == 1:
            current_expand += 1
            current_contract = 0
        elif df['range_contracting'].iloc[i] == 1:
            current_contract += 1
            current_expand = 0
        else:
            current_expand = 0
            current_contract = 0
        
        df.loc[df.index[i], 'expanding_streak'] = current_expand
        df.loc[df.index[i], 'contracting_streak'] = current_contract
    
    # Exponential decay on streak length
    df['streak_weight'] = np.exp(-0.2 * np.maximum(df['expanding_streak'], df['contracting_streak']))
    
    # Combine all components multiplicatively
    df['factor'] = (df['range_percentile'] * df['momentum_reversal'] * 
                    df['volume_deviation'] * df['streak_weight'])
    
    # Handle any remaining NaN values
    df['factor'] = df['factor'].fillna(0)
    
    return df['factor']
