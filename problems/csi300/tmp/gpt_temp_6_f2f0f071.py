import pandas as pd
def heuristics_v2(df):
    # Define momentum period (21 days)
    momentum_window = 21
    
    # Calculate cumulative return over the momentum window
    # Using shift(1) to avoid lookahead bias (only past data)
    close_prices = df['close']
    past_close = close_prices.shift(momentum_window)
    cumulative_return = (close_prices - past_close) / past_close
    
    # Normalize momentum by window length
    normalized_momentum = cumulative_return / momentum_window
    
    # Rank securities based on normalized momentum
    # Using pandas qcut to assign decile ranks (1=lowest, 10=highest momentum)
    factor = normalized_momentum.groupby(level=0).transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates='drop') + 1
    )
    
    return factor
