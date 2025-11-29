import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate intraday prices (assuming first hour = 10:00, midday = 13:00, last hour = 15:00)
    # In practice, these would come from intraday data, but for daily data we'll use approximations
    data['first_hour_high'] = data['high'].rolling(window=5, min_periods=1).apply(lambda x: x[:1].max() if len(x) >= 1 else np.nan)
    data['first_hour_low'] = data['low'].rolling(window=5, min_periods=1).apply(lambda x: x[:1].min() if len(x) >= 1 else np.nan)
    data['first_hour_close'] = data['close'].rolling(window=5, min_periods=1).apply(lambda x: x[0] if len(x) >= 1 else np.nan)
    
    data['last_hour_high'] = data['high'].rolling(window=5, min_periods=1).apply(lambda x: x[-1:].max() if len(x) >= 1 else np.nan)
    data['last_hour_low'] = data['low'].rolling(window=5, min_periods=1).apply(lambda x: x[-1:].min() if len(x) >= 1 else np.nan)
    
    # Midday price approximation (average of open and midday high/low)
    data['midday_price'] = (data['open'] + data['high'].rolling(window=3, min_periods=1).mean()) / 2
    
    # Volume approximations (assuming first hour = 25% of daily, last hour = 25% of daily)
    data['first_hour_volume'] = data['volume'] * 0.25
    data['last_hour_volume'] = data['volume'] * 0.25
    
    # Intraday Price Momentum Structure
    data['morning_momentum'] = (data['first_hour_close'] - data['open']) / (data['first_hour_high'] - data['first_hour_low'] + 1e-8)
    data['afternoon_momentum'] = (data['close'] - data['midday_price']) / (data['last_hour_high'] - data['last_hour_low'] + 1e-8)
    data['intraday_momentum_ratio'] = data['morning_momentum'] / (data['afternoon_momentum'] + 1e-8)
    
    # Volume Concentration Patterns
    data['morning_volume_focus'] = data['first_hour_volume'] / (data['volume'] + 1e-8)
    data['afternoon_volume_focus'] = data['last_hour_volume'] / (data['volume'] + 1e-8)
    data['volume_skew'] = data['morning_volume_focus'] - data['afternoon_volume_focus']
    
    # Momentum-Volume Alignment
    data['morning_alignment'] = data['morning_momentum'] * data['morning_volume_focus']
    data['afternoon_alignment'] = data['afternoon_momentum'] * data['afternoon_volume_focus']
    data['alignment_divergence'] = data['morning_alignment'] - data['afternoon_alignment']
    
    # Multi-Day Persistence
    data['momentum_persistence'] = data['intraday_momentum_ratio'] * data['intraday_momentum_ratio'].shift(1)
    data['volume_persistence'] = data['volume_skew'] * data['volume_skew'].shift(1)
    data['persistence_convergence'] = data['momentum_persistence'] * data['volume_persistence']
    
    # Adaptive Alpha Construction
    data['base_signal'] = data['alignment_divergence'] * data['intraday_momentum_ratio']
    data['persistence_enhancement'] = data['base_signal'] * data['persistence_convergence']
    data['final_alpha'] = data['persistence_enhancement'] * data['volume_skew']
    
    # Return the final alpha factor
    return data['final_alpha']
