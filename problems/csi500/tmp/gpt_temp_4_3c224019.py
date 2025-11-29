import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel cross-sectional alpha factor combining gap-range efficiency with regime transitions,
    amount-range efficiency with volume divergence, momentum transfer with range flow,
    and volatility-weighted efficiency with momentum confirmation.
    """
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Pre-calculate common components
    data['prev_close'] = data.groupby(level=1)['close'].shift(1)
    data['daily_return'] = data.groupby(level=1)['close'].pct_change()
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['daily_range'] = data['high'] - data['low']
    data['intraday_move'] = data['close'] - data['open']
    data['range_efficiency'] = np.abs(data['intraday_move']) / np.where(data['daily_range'] > 0, data['daily_range'], np.nan)
    
    # Gap-Range Efficiency with Regime Transition
    # Gap persistence strength
    data['gap_persistence'] = np.where(
        data['overnight_gap'] * data['intraday_move'] > 0,
        np.abs(data['intraday_move']) / np.abs(data['overnight_gap']),
        0
    )
    
    # Range confirmation score
    data['range_confirmation'] = np.where(
        data['overnight_gap'] * data['intraday_move'] > 0,
        data['range_efficiency'],
        -data['range_efficiency']
    )
    
    # Price-volume correlation regimes
    data['ret_3d'] = data.groupby(level=1)['close'].pct_change(3)
    data['vol_3d'] = data.groupby(level=1)['volume'].rolling(3).mean().reset_index(level=0, drop=True)
    data['ret_10d'] = data.groupby(level=1)['close'].pct_change(10)
    data['vol_10d'] = data.groupby(level=1)['volume'].rolling(10).mean().reset_index(level=0, drop=True)
    
    # Rolling correlations
    data['corr_3d'] = data.groupby(level=1).apply(
        lambda x: x['daily_return'].rolling(3).corr(x['volume'])
    ).reset_index(level=0, drop=True)
    
    data['corr_10d'] = data.groupby(level=1).apply(
        lambda x: x['daily_return'].rolling(10).corr(x['volume'])
    ).reset_index(level=0, drop=True)
    
    # Regime transition strength
    data['corr_regime_change'] = data['corr_3d'] - data['corr_10d']
    data['gap_regime_alignment'] = data['overnight_gap'] * data['corr_regime_change']
    
    # Gap-range regime factor
    data['gap_range_regime'] = data['range_confirmation'] * data['gap_regime_alignment'] * data['gap_persistence']
    
    # Amount-Range Efficiency with Volume Divergence
    data['amount_efficiency'] = data['daily_return'] / np.where(data['amount'] > 0, data['amount'], np.nan)
    data['amount_range_interaction'] = data['amount_efficiency'] * data['range_efficiency']
    
    # Volume divergence strength
    data['volume_divergence'] = data['corr_regime_change'] * np.sign(data['daily_return'])
    data['amount_range_divergence'] = data['amount_range_interaction'] * data['volume_divergence']
    
    # Opening-Closing Momentum Transfer with Range Flow
    data['opening_momentum'] = data['overnight_gap']
    
    # Momentum transfer efficiency (simplified)
    mid_price = (data['high'] + data['low']) / 2
    data['morning_momentum'] = (mid_price - data['open']) / data['open']
    data['afternoon_momentum'] = (data['close'] - mid_price) / mid_price
    
    data['momentum_transfer'] = np.where(
        data['opening_momentum'] * data['afternoon_momentum'] > 0,
        np.abs(data['afternoon_momentum']) / (np.abs(data['opening_momentum']) + 1e-8),
        0
    )
    
    # Range flow measure
    morning_range = mid_price - data['open']
    afternoon_range = data['close'] - mid_price
    data['range_flow'] = np.where(
        np.abs(morning_range) + np.abs(afternoon_range) > 0,
        (np.abs(afternoon_range) - np.abs(morning_range)) / (np.abs(afternoon_range) + np.abs(morning_range)),
        0
    )
    
    data['momentum_range_flow'] = data['momentum_transfer'] * data['range_flow']
    
    # Volatility-Weighted Efficiency with Momentum Confirmation
    data['volatility_20d'] = data.groupby(level=1)['daily_return'].rolling(20).std().reset_index(level=0, drop=True)
    data['return_5d'] = data.groupby(level=1)['close'].pct_change(5)
    data['return_20d'] = data.groupby(level=1)['close'].pct_change(20)
    
    # Combined efficiency score
    data['combined_efficiency'] = data['amount_efficiency'] * data['range_efficiency']
    
    # Volatility regime adjustment
    vol_median = data.groupby(level=0)['volatility_20d'].median()
    data['vol_regime'] = data['volatility_20d'] / vol_median
    
    # Volatility-weighted efficiency
    data['vol_weighted_efficiency'] = data['combined_efficiency'] / (data['vol_regime'] + 1e-8)
    
    # Momentum confirmation
    data['momentum_confirmation'] = np.sign(data['return_5d']) * np.sign(data['return_20d'])
    data['vol_efficiency_momentum'] = data['vol_weighted_efficiency'] * data['momentum_confirmation']
    
    # Final factor combination (equal weighted for simplicity)
    factors = [
        'gap_range_regime',
        'amount_range_divergence', 
        'momentum_range_flow',
        'vol_efficiency_momentum'
    ]
    
    # Z-score normalization within each day
    for factor in factors:
        data[f'{factor}_z'] = data.groupby(level=0)[factor].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
    
    # Equal weighted combination of normalized factors
    data['final_factor'] = (
        data['gap_range_regime_z'] + 
        data['amount_range_divergence_z'] + 
        data['momentum_range_flow_z'] + 
        data['vol_efficiency_momentum_z']
    ) / 4
    
    return data['final_factor']
