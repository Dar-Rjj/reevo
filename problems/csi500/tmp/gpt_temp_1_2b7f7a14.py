import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price features
    data['returns'] = data['close'].pct_change()
    data['hl_range'] = (data['high'] - data['low']) / data['close']
    data['oc_range'] = (data['close'] - data['open']) / data['open']
    
    # Intraday Momentum Divergence Factor
    # Estimate 30-min VWAP using first 30 minutes of data (assuming 6.5 hour trading day)
    data['vwap_30min'] = (data['high'] + data['low'] + data['close']) / 3  # Simplified VWAP proxy
    data['early_momentum'] = (data['vwap_30min'] - data['open']) / data['open']
    
    # Mid-session momentum (30-min to 2-hour)
    data['mid_momentum'] = data['returns'].rolling(window=3).mean()  # Proxy for mid-session
    
    # Late session momentum (last 2 hours)
    data['late_momentum'] = (data['close'] - data['open'].shift(1)) / data['open'].shift(1)  # Previous day's open to today's close
    
    # Momentum divergence signal
    data['momentum_divergence'] = (
        np.sign(data['early_momentum']) + 
        np.sign(data['mid_momentum']) + 
        np.sign(data['late_momentum'])
    )
    
    # Volume confirmation
    data['volume_ma'] = data['volume'].rolling(window=5).mean()
    data['volume_confirmation'] = data['volume'] / data['volume_ma']
    data['momentum_factor'] = data['momentum_divergence'] * data['volume_confirmation']
    
    # Volume-Clustered Price Impact
    data['volume_quantile'] = data['volume'].rolling(window=20).apply(
        lambda x: pd.qcut(x, 2, labels=False, duplicates='drop').iloc[-1] if len(x) == 20 else np.nan, 
        raw=False
    )
    
    # Price impact for high vs low volume periods
    data['high_vol_impact'] = np.where(
        data['volume_quantile'] == 1, 
        data['returns'].abs() / data['volume'], 
        0
    )
    data['low_vol_impact'] = np.where(
        data['volume_quantile'] == 0, 
        data['returns'].abs() / (data['volume'] + 1e-8), 
        0
    )
    
    data['volume_impact_factor'] = (
        data['high_vol_impact'].rolling(window=10).mean() - 
        data['low_vol_impact'].rolling(window=10).mean()
    )
    
    # Opening Auction Imbalance
    data['gap_magnitude'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['auction_strength'] = data['gap_magnitude'] * data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Multi-Scale Volatility Regime
    data['micro_vol'] = data['hl_range'].rolling(window=2).std()
    data['meso_vol'] = data['returns'].abs().rolling(window=5).mean()
    data['macro_vol'] = data['returns'].abs().rolling(window=20).mean()
    
    data['volatility_regime'] = (
        data['micro_vol'].rank(pct=True) + 
        data['meso_vol'].rank(pct=True) + 
        data['macro_vol'].rank(pct=True)
    ) / 3
    
    # Price Path Fractality
    data['fractal_1'] = data['returns'].abs().rolling(window=3).std()
    data['fractal_2'] = data['returns'].abs().rolling(window=5).std()
    data['fractal_3'] = data['returns'].abs().rolling(window=8).std()
    
    data['fractal_similarity'] = (
        (data['fractal_1'] / (data['fractal_2'] + 1e-8)) + 
        (data['fractal_2'] / (data['fractal_3'] + 1e-8))
    ) / 2
    
    # Amount-Concentrated Flow
    data['avg_trade_size'] = data['amount'] / (data['volume'] + 1e-8)
    data['large_block_ratio'] = data['avg_trade_size'].rolling(window=10).apply(
        lambda x: (x > x.quantile(0.8)).sum() / len(x), raw=False
    )
    
    data['flow_quality'] = data['large_block_ratio'] * data['returns'].abs()
    
    # Combine all factors
    factors = pd.DataFrame({
        'momentum': data['momentum_factor'],
        'volume_impact': data['volume_impact_factor'],
        'auction': data['auction_strength'],
        'vol_regime': data['volatility_regime'],
        'fractal': data['fractal_similarity'],
        'flow': data['flow_quality']
    })
    
    # Remove any infinite values and fill NaN
    factors = factors.replace([np.inf, -np.inf], np.nan)
    factors = factors.fillna(method='ffill').fillna(0)
    
    # Z-score normalization for each factor
    factors_z = factors.apply(lambda x: (x - x.rolling(window=60, min_periods=1).mean()) / 
                             (x.rolling(window=60, min_periods=1).std() + 1e-8))
    
    # Equal-weighted combination
    final_factor = factors_z.mean(axis=1)
    
    return final_factor
