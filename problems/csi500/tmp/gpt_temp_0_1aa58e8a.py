import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor combining momentum persistence, intraday reversal asymmetry,
    price-volume efficiency, range dynamics, and liquidity momentum divergence.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Feature 1: Cross-Sectional Momentum Persistence
    # 3-day momentum acceleration with volume divergence
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_accel'] = data['momentum_3d'] - data['momentum_3d'].shift(3)
    
    # Volume momentum and divergence
    data['volume_ma'] = data['volume'].rolling(window=5).mean()
    data['volume_momentum'] = data['volume'] / data['volume_ma'] - 1
    data['volume_accel'] = data['volume_momentum'] - data['volume_momentum'].shift(3)
    
    # Price-volume acceleration divergence
    data['pv_divergence'] = data['momentum_accel'] - data['volume_accel']
    
    # Feature 2: Intraday Reversal Asymmetry
    # Morning-afternoon divergence
    data['morning_range'] = (data['high'] - data['low']) / data['open']
    data['afternoon_momentum'] = (data['close'] - data['open']) / data['open']
    data['intraday_divergence'] = data['morning_range'] - abs(data['afternoon_momentum'])
    
    # Volume concentration (intraday proxy)
    data['volume_std_5d'] = data['volume'].rolling(window=5).std()
    data['volume_concentration'] = data['volume'] / data['volume_std_5d']
    
    # Feature 3: Price-Volume Fractal Efficiency
    # Multi-scale inefficiency detection
    data['daily_range'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['range_efficiency'] = data['daily_range'].rolling(window=5).std()
    
    # Volume fractal properties
    data['volume_rank'] = data['volume'].rolling(window=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Feature 4: Dynamic Range Compression-Expansion
    # Range state detection
    data['range_ma'] = data['daily_range'].rolling(window=10).mean()
    data['range_compression'] = data['daily_range'] / data['range_ma']
    
    # Volume-volatility coupling
    data['volume_range_corr'] = data['volume'].rolling(window=5).corr(data['daily_range'])
    
    # Feature 5: Liquidity Momentum Divergence
    # Amount-based momentum
    data['amount_ma'] = data['amount'].rolling(window=5).mean()
    data['amount_momentum'] = data['amount'] / data['amount_ma'] - 1
    
    # Price-liquidity divergence
    data['price_liquidity_div'] = data['momentum_3d'] - data['amount_momentum']
    
    # Combine features into final factor
    # Normalize each component before combination
    features = [
        'pv_divergence', 'intraday_divergence', 'volume_concentration',
        'range_efficiency', 'volume_rank', 'range_compression',
        'volume_range_corr', 'price_liquidity_div'
    ]
    
    # Z-score normalization within each day (cross-sectional)
    factor_values = pd.Series(index=data.index, dtype=float)
    
    for date in data.index:
        day_data = data.loc[date]
        if len(day_data.shape) == 1:  # Single stock case
            day_data = pd.DataFrame([day_data])
        
        # Calculate weighted combination
        scores = []
        for feature in features:
            if feature in day_data.columns:
                vals = day_data[feature].fillna(0)
                if len(vals) > 1:  # Cross-sectional normalization
                    z_scores = (vals - vals.mean()) / (vals.std() + 1e-8)
                else:
                    z_scores = vals * 0  # Neutral if only one stock
                scores.append(z_scores)
        
        if scores:
            # Equal weighted combination of normalized features
            combined_score = sum(scores) / len(scores)
            if len(combined_score) == 1:
                factor_values.loc[date] = combined_score.iloc[0]
            else:
                factor_values.loc[date] = combined_score.mean()
        else:
            factor_values.loc[date] = 0
    
    # Final smoothing and normalization
    factor_values = factor_values.rolling(window=3, min_periods=1).mean()
    factor_values = (factor_values - factor_values.rolling(window=20, min_periods=1).mean()) / (factor_values.rolling(window=20, min_periods=1).std() + 1e-8)
    
    return factor_values
