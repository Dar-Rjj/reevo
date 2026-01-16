import pandas as pd
import numpy as np
def heuristics_v2(df):
    import pandas as pd
    import numpy as np

    # Range Signal
    df['current_range'] = (df['high'] - df['low']) / ((df['open'] + df['close']) / 2)
    df['ma_range'] = df['high'].rolling(window=5).mean() - df['low'].rolling(window=5).mean()
    df['range_signal'] = df['current_range'] / df['ma_range']

    # Momentum Signal
    df['ma_close'] = df['close'].rolling(window=5).mean()
    df['price_strength'] = df['close'] / df['ma_close']
    df['momentum_change'] = df['close'] - df['close'].shift(5)
    df['sma_momentum'] = df['momentum_change'].rolling(window=3).mean()
    df['momentum_signal'] = df['price_strength'] * df['sma_momentum']

    # Volume Signal
    df['ma_volume'] = df['volume'].rolling(window=5).mean()
    df['volume_spike'] = df['volume'] / df['ma_volume']
    df['volume_slope'] = df['volume'].rolling(window=3).apply(lambda x: np.polyfit(range(3), x, 1)[0])
    df['volume_signal'] = df['volume_spike'] * df['volume_slope']

    # Composite Construction
    df['range_momentum_ratio'] = df['range_signal'] * df['momentum_signal']
    df['volume_contribution'] = df['volume_signal']
    df['volatility'] = df['close'].pct_change().rolling(window=5).std()

    # Final Factor
    df['final_factor'] = (df['range_momentum_ratio'] * df['volume_contribution']) / df['volatility']
    df['final_factor'] = (df['final_factor'] - df['final_factor'].mean()) / df['final_factor'].std()

    return df['final_factor']
