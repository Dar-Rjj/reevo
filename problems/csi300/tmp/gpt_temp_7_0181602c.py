import pandas as pd
import pandas as pd

def heuristics_v2(data):
    # Initialize output series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Sentiment Strength branch
    # Volume normalization with 20-day rolling mean
    volume_rolling_mean = data['volume'].rolling(window=20, min_periods=1).mean()
    normalized_volume = data['volume'] / volume_rolling_mean
    
    # Volatility adjustment with 30-day rolling std of high prices
    high_rolling_std = data['high'].rolling(window=30, min_periods=1).std()
    volatility_adjustment = 1 / (1 + high_rolling_std)
    
    sentiment_strength = normalized_volume * volatility_adjustment
    
    # Price Momentum branch
    # Delta between current close and 5-day ago close
    delta_close = data['close'] - data['close'].shift(5)
    
    # Calculate returns (daily percentage change)
    returns = data['close'].pct_change()
    
    # Rolling rank of returns over 10-day window
    ranked_returns = returns.rolling(window=10, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    price_momentum = delta_close * ranked_returns
    
    # Combine both branches
    factor = sentiment_strength + price_momentum
    
    return factor
