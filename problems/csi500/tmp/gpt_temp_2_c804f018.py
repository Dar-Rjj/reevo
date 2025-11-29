import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate required intermediate series
    data['range'] = data['high'] - data['low']
    data['prev_close'] = data['close'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    
    # Intraday Momentum Persistence (IMP)
    data['normalized_momentum'] = (data['close'] - data['low']) / np.where(data['range'] > 0, data['range'], np.nan)
    
    # Calculate 5-day momentum autocorrelation using rolling window
    imp_factor = pd.Series(index=data.index, dtype=float)
    for i in range(5, len(data)):
        window_data = data['normalized_momentum'].iloc[i-5:i+1]
        if window_data.notna().sum() >= 4:  # Require at least 4 non-NaN values
            imp_factor.iloc[i] = window_data.autocorr(lag=1)
    
    # Gap Filling Efficiency (GFE)
    data['overnight_gap'] = (data['open'] / data['prev_close']) - 1
    data['gap_to_range_ratio'] = np.abs(data['overnight_gap']) / np.where(data['range'] > 0, data['range'], np.nan)
    gfe_factor = -data['gap_to_range_ratio']  # Negative because larger gaps that don't fill are inefficient
    
    # Volume-Confirmed Efficiency (VCE)
    data['range_efficiency'] = np.abs(data['close'] - data['prev_close']) / np.where(data['range'] > 0, data['range'], np.nan)
    data['log_volume'] = np.log(data['volume'].replace(0, np.nan))
    vce_factor = data['range_efficiency'] * data['log_volume']
    
    # Price-Amount Trend Consistency (PATC)
    data['price_trend'] = np.sign(data['close'] - data['prev_close'])
    data['amount_trend'] = np.sign(data['amount'] - data['prev_amount'])
    patc_factor = data['price_trend'] * data['amount_trend']
    
    # Volatility-Regime Reversal (VRR)
    # Calculate 10-day range volatility
    data['daily_range'] = data['high'] - data['low']
    range_volatility = data['daily_range'].rolling(window=10, min_periods=8).std()
    
    # Calculate previous day return reversal efficiency
    data['prev_return'] = data['close'].pct_change(1)
    vrr_factor = -data['prev_return'] / np.where(range_volatility > 0, range_volatility, np.nan)
    
    # Combine factors with equal weights
    factors_df = pd.DataFrame({
        'IMP': imp_factor,
        'GFE': gfe_factor,
        'VCE': vce_factor,
        'PATC': patc_factor,
        'VRR': vrr_factor
    })
    
    # Z-score normalization for each factor cross-sectionally
    for col in factors_df.columns:
        factors_df[col] = (factors_df[col] - factors_df[col].mean()) / factors_df[col].std()
    
    # Equal-weighted combination
    factor = factors_df.mean(axis=1)
    
    return factor
