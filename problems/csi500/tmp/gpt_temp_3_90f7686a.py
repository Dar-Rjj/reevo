import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # High-Low Volatility Adjusted Intraday Momentum
    intraday_momentum = (data['high'] - data['open']) + (data['close'] - data['low'])
    high_low_volatility = (data['high'] - data['low']) / data['close']
    # Avoid division by zero
    high_low_volatility = high_low_volatility.replace(0, np.nan)
    factor1 = intraday_momentum / high_low_volatility * data['volume']
    
    # Price Gap Reversal with Volume Confirmation
    prev_close = data['close'].shift(1)
    overnight_gap = (data['open'] - prev_close) / prev_close
    volume_ma_5 = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_trend = data['volume'] / volume_ma_5
    factor2 = -overnight_gap * volume_trend * (data['high'] - data['low'])
    
    # Amplitude-Weighted Price Change Persistence
    price_amplitude = (data['high'] - data['low']) / data['open']
    
    # Calculate 3-day price moves and persistence count
    price_change_3d = data['close'].pct_change(periods=3)
    direction = np.sign(price_change_3d)
    
    # Count consecutive same-direction moves
    persistence_count = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if direction.iloc[i] == direction.iloc[i-1] and not pd.isna(direction.iloc[i]) and not pd.isna(direction.iloc[i-1]):
            persistence_count.iloc[i] = persistence_count.iloc[i-1] + 1
    
    factor3 = price_amplitude * persistence_count * data['amount']
    
    # Volume-Driven Opening Price Efficiency
    opening_efficiency = (data['open'] - prev_close) / (data['high'] - data['low'])
    # Avoid division by zero
    opening_efficiency = opening_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Since we don't have first hour volume, use opening hour approximation
    # Assuming first hour volume correlates with opening amount
    opening_amount_ratio = data['amount'].rolling(window=5, min_periods=1).apply(
        lambda x: x.iloc[0] / x.mean() if x.mean() > 0 else np.nan
    )
    momentum_enhanced = opening_efficiency * opening_amount_ratio * (data['close'] - prev_close) / data['close']
    factor4 = momentum_enhanced
    
    # Relative Strength with Volume Divergence
    # Since we don't have sector data, use market-relative approach
    stock_return_5d = data['close'].pct_change(periods=5)
    market_return_5d = data['close'].pct_change(periods=5).rolling(window=20, min_periods=1).mean()
    
    stock_volume_change = data['volume'].pct_change(periods=5)
    market_volume_change = data['volume'].pct_change(periods=5).rolling(window=20, min_periods=1).mean()
    
    relative_strength = stock_return_5d / market_return_5d
    volume_divergence = stock_volume_change / market_volume_change
    
    # Avoid division issues
    relative_strength = relative_strength.replace([np.inf, -np.inf], np.nan)
    volume_divergence = volume_divergence.replace([np.inf, -np.inf], np.nan)
    
    factor5 = relative_strength * volume_divergence * data['amount'] * data['volume']
    
    # Combine factors with equal weights
    combined_factor = (factor1.fillna(0) + factor2.fillna(0) + factor3.fillna(0) + 
                      factor4.fillna(0) + factor5.fillna(0)) / 5
    
    return combined_factor
