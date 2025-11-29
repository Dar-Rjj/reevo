import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Momentum Divergence
    # AM Momentum = (Close - Open) / (High - Low)
    am_momentum = (data['close'] - data['open']) / (data['high'] - data['low'])
    am_momentum = am_momentum.replace([np.inf, -np.inf], np.nan)
    
    # PM Momentum = (Close - (High + Low)/2) / (High - Low)
    pm_momentum = (data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low'])
    pm_momentum = pm_momentum.replace([np.inf, -np.inf], np.nan)
    
    # Momentum Divergence = |AM Momentum - PM Momentum| × rolling_correlation(AM Momentum, PM Momentum, 3)
    momentum_diff = abs(am_momentum - pm_momentum)
    
    # Calculate rolling correlation using only past data
    rolling_corr = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 2:  # Need at least 3 points for correlation
            window_am = am_momentum.iloc[i-2:i+1]
            window_pm = pm_momentum.iloc[i-2:i+1]
            if not window_am.isna().any() and not window_pm.isna().any():
                rolling_corr.iloc[i] = window_am.corr(window_pm)
            else:
                rolling_corr.iloc[i] = 0
        else:
            rolling_corr.iloc[i] = 0
    
    momentum_divergence = momentum_diff * rolling_corr
    
    # Medium-Term Trend
    # Price Return = (Close - Close_5d_ago) / Close_5d_ago
    close_5d_ago = data['close'].shift(5)
    price_return = (data['close'] - close_5d_ago) / close_5d_ago
    
    # Volume Trend = ln(Volume / Volume_5d_ago)
    volume_5d_ago = data['volume'].shift(5)
    volume_trend = np.log(data['volume'] / volume_5d_ago)
    volume_trend = volume_trend.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volume Acceleration
    # Volume Flow Direction = sign((Close - Open) × Volume)
    volume_flow_direction = np.sign((data['close'] - data['open']) * data['volume'])
    
    # Volume Acceleration = (Volume / rolling_mean(Volume, 3)) × Volume Flow Direction
    volume_rolling_mean = data['volume'].rolling(window=3, min_periods=1).mean()
    volume_acceleration = (data['volume'] / volume_rolling_mean) * volume_flow_direction
    
    # Price Position Momentum
    # Position = (Close - Low)/(High - Low)
    position = (data['close'] - data['low']) / (data['high'] - data['low'])
    position = position.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    
    # Position Momentum = rolling_mean((Position - Position_1d_ago), 5)
    position_1d_ago = position.shift(1)
    position_change = position - position_1d_ago
    position_momentum = position_change.rolling(window=5, min_periods=1).mean()
    
    # Factor Synthesis
    # Momentum-Trend Divergence = (Momentum Divergence - Price Return) × Volume Trend
    momentum_trend_divergence = (momentum_divergence - price_return) * volume_trend
    
    # Final Factor = Momentum-Trend Divergence × Volume Acceleration × Position Momentum
    final_factor = momentum_trend_divergence * volume_acceleration * position_momentum
    
    return final_factor
