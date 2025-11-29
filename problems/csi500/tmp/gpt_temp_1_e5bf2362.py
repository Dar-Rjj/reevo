import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional intraday microstructure alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Calculate basic metrics
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    
    # Opening Auction Imbalance Analysis
    data['opening_gap_efficiency'] = ((data['open'] - data['prev_close']) / 
                                    (data['high'] - data['low']).replace(0, np.nan) * 
                                    (data['volume'].fillna(0) / data['volume'].rolling(5).mean()))
    
    data['pre_market_pressure'] = ((data['open'] - data['prev_low']) / 
                                 (data['prev_high'] - data['prev_low']).replace(0, np.nan) * 
                                 data['volume'])
    
    # Volume-Weighted Price Analysis
    data['vwap'] = (data['amount'] / data['volume']).replace([np.inf, -np.inf], np.nan)
    
    data['high_volume_node'] = ((data['vwap'] - data['open']) / 
                              (data['high'] - data['low']).replace(0, np.nan))
    
    data['volume_profile_skew'] = ((data['high'] - data['low']) / 
                                 (data['high'] - data['low']).replace(0, np.nan))
    
    # Session Boundary Effects
    data['overnight_gap'] = (abs(data['open'] - data['prev_close']) / 
                           (data['high'] - data['low']).replace(0, np.nan))
    
    data['gap_fill_probability'] = ((data['high'] - data['prev_close']) / 
                                  (data['prev_close'] - data['prev_low']).replace(0, np.nan))
    
    # Price Discovery Efficiency
    data['intraday_efficiency'] = (abs(data['close'] - data['open']) / 
                                 (data['high'] - data['low']).replace(0, np.nan))
    
    data['microstructure_friction'] = (abs(data['close'] - data['vwap']) / 
                                     (data['high'] - data['low']).replace(0, np.nan))
    
    # Volume-Time Analysis
    data['volume_intensity'] = data['volume'] / data['volume'].rolling(10).mean()
    
    # Composite Microstructure Score
    factors = [
        'opening_gap_efficiency',
        'pre_market_pressure', 
        'high_volume_node',
        'volume_profile_skew',
        'overnight_gap',
        'gap_fill_probability',
        'intraday_efficiency',
        'microstructure_friction',
        'volume_intensity'
    ]
    
    # Normalize each factor cross-sectionally
    alpha_scores = pd.Series(index=data.index, dtype=float)
    
    for date in data.index:
        day_data = data.loc[date]
        day_score = 0
        
        for factor in factors:
            if factor in day_data and pd.notna(day_data[factor]):
                # Simple rank-based scoring
                if isinstance(day_data, pd.Series):
                    # Single stock case
                    day_score += day_data[factor]
                else:
                    # Cross-sectional ranking
                    valid_data = day_data[factor].dropna()
                    if len(valid_data) > 1:
                        ranks = valid_data.rank(pct=True)
                        day_score += ranks.mean() if len(ranks) > 0 else 0
        
        alpha_scores.loc[date] = day_score
    
    # Final smoothing and normalization
    alpha_scores = alpha_scores.rolling(3, min_periods=1).mean()
    alpha_scores = (alpha_scores - alpha_scores.rolling(20).mean()) / alpha_scores.rolling(20).std()
    
    return alpha_scores.fillna(0)
