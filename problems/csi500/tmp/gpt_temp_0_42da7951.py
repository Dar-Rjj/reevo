import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Intraday Return Signal
    denominator = data['open'] + data['close']
    intraday_return_signal = np.where(
        denominator != 0,
        (data['high'] - data['low']) / denominator * 2,
        0
    )
    
    # Compute Volume-Price Interaction
    volume_price_interaction = (data['close'] - data['open']) * data['volume']
    
    # Calculate 5-day rolling standard deviation of Volume-Price Interaction
    volume_price_std = volume_price_interaction.rolling(window=5, min_periods=3).std()
    
    # Generate Combined Factor
    # Calculate 3-day lagged correlation between Intraday Return Signal and Volume-Price Interaction
    correlation_window = 3
    
    def rolling_correlation(x, y):
        correlations = []
        for i in range(len(x)):
            if i < correlation_window:
                correlations.append(np.nan)
            else:
                window_x = x[i-correlation_window:i]
                window_y = y[i-correlation_window:i]
                if len(window_x) == correlation_window and len(window_y) == correlation_window:
                    corr = np.corrcoef(window_x, window_y)[0, 1]
                    correlations.append(corr if not np.isnan(corr) else 0)
                else:
                    correlations.append(0)
        return pd.Series(correlations, index=x.index)
    
    lagged_correlation = rolling_correlation(
        pd.Series(intraday_return_signal, index=data.index),
        pd.Series(volume_price_interaction, index=data.index)
    )
    
    # Calculate Volume/Amount ratio
    volume_amount_ratio = np.where(
        data['amount'] != 0,
        data['volume'] / data['amount'],
        0
    )
    
    # Generate final factor
    factor = lagged_correlation * volume_amount_ratio
    
    return factor
