import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic metrics
    data['range'] = data['high'] - data['low']
    data['prev_range'] = data['range'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_close'] = data['close'].shift(1)
    data['price_change'] = data['close'] - data['prev_close']
    data['intraday_return'] = data['close'] - data['open']
    data['daily_return'] = (data['close'] - data['prev_close']) / data['prev_close']
    
    # Volatility State Momentum
    data['low_to_high_transition'] = ((data['range'] - data['prev_range']) / data['prev_range'].replace(0, np.nan)) * data['volume']
    data['high_to_low_decay'] = -((data['range'] - data['prev_range']) / data['prev_range'].replace(0, np.nan)) * data['amount']
    
    # Liquidity Regime Adaptation
    data['thin_to_thick_efficiency'] = ((data['volume'] - data['prev_volume']) / data['prev_volume'].replace(0, np.nan)) * data['intraday_return']
    
    # Calculate spread regime persistence
    data['range_direction'] = np.where(data['range'] > data['prev_range'], 1, 
                                      np.where(data['range'] < data['prev_range'], -1, 0))
    data['spread_persistence'] = data.groupby(data['range_direction'].ne(data['range_direction'].shift()).cumsum())['range_direction'].cumcount() * data['price_change']
    
    # Multi-timeframe Anchor Effects
    data['weekly_level'] = data['close'].rolling(window=5, min_periods=1).mean()
    data['weekly_vs_daily'] = ((data['close'] - data['weekly_level']) / data['range'].replace(0, np.nan)) * data['volume']
    
    # Historical Support/Resistance
    data['historical_level'] = data['close'].rolling(window=20, min_periods=1).mean()
    data['historical_reactivation'] = ((data['close'] - data['historical_level']) / data['range'].replace(0, np.nan)) * data['volume']
    
    # Volume-Price Patterns
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_ma_10'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['micro_macro_volume'] = (data['volume'] / data['volume_ma_5']) * data['range'] * data['price_change']
    data['volume_scaling'] = (data['volume_ma_5'] / data['volume_ma_10']) * data['intraday_return']
    
    # Momentum Phase Transition
    data['prev_return'] = data['daily_return'].shift(1)
    data['momentum_inflection'] = ((data['daily_return'] - data['prev_return']) / data['range'].replace(0, np.nan)) * data['volume']
    
    # Range Expansion-Contraction
    data['avg_range'] = data['range'].rolling(window=10, min_periods=1).mean()
    data['range_breakout_prob'] = (data['range'] / data['avg_range']) * ((data['volume'] - data['prev_volume']) / data['prev_volume'].replace(0, np.nan))
    
    # Range Boundary Behavior
    data['boundary_attraction'] = ((data['close'] - data['prev_close']) / data['range'].replace(0, np.nan)) * data['volume']
    
    # Volume Concentration
    data['volume_change'] = (data['volume'] - data['prev_volume']) / data['prev_volume'].replace(0, np.nan)
    data['volume_accumulation'] = (data['volume_change'] / data['price_change'].replace(0, np.nan)) * data['range']
    
    # Price Discovery Efficiency
    data['gap_absorption'] = ((data['high'] - data['open']) / (data['open'] - data['prev_close']).replace(0, np.nan)) * data['volume']
    data['price_impact'] = ((data['close'] - data['prev_close']) / data['range'].replace(0, np.nan)) * data['volume']
    
    # Intraday Reversal with Overnight Gap
    data['intraday_reversal'] = (data['close'] - data['open']) / data['range'].replace(0, np.nan)
    data['overnight_gap'] = abs(data['open'] - data['prev_close']) / data['prev_close'].replace(0, np.nan)
    data['volume_median_20'] = data['volume'].rolling(window=20, min_periods=1).median()
    data['volume_confirmation'] = data['volume'] / data['volume_median_20']
    
    # Combined signals
    data['signal_intraday_reversal'] = -data['intraday_reversal'] * data['overnight_gap'] * (data['volume'] / data['volume'].rolling(window=5, min_periods=1).sum()) * data['amount']
    
    data['range_efficiency'] = (data['close'] - data['open']) / data['range'].replace(0, np.nan)
    data['return_3d'] = data['close'].pct_change(3)
    data['return_5d'] = data['close'].pct_change(5)
    data['momentum_divergence'] = data['return_3d'] - data['return_5d']
    data['signal_range_momentum'] = data['range_efficiency'] * data['momentum_divergence'] * data['volume_confirmation'] * data['range']
    
    # Volatility-Adjusted Opening Jump
    data['opening_jump'] = (data['open'] - data['prev_close']) / data['prev_close'].replace(0, np.nan)
    data['volatility_adj'] = data['range'] / data['open'].replace(0, np.nan)
    data['jump_sign'] = np.sign(data['opening_jump'])
    data['persistence_count'] = data.groupby(data['jump_sign'].ne(data['jump_sign'].shift()).cumsum())['jump_sign'].cumcount()
    data['signal_volatility_jump'] = data['opening_jump'] * data['volatility_adj'] * data['persistence_count'] * data['volume']
    
    # Amount-Driven Price Rejection
    data['upper_rejection'] = (data['high'] - data['close']) / data['range'].replace(0, np.nan)
    data['lower_rejection'] = (data['close'] - data['low']) / data['range'].replace(0, np.nan)
    data['amount_ma_10'] = data['amount'].rolling(window=10, min_periods=1).mean()
    data['amount_intensity'] = data['amount'] / data['amount_ma_10']
    data['signal_amount_rejection'] = (data['upper_rejection'] - data['lower_rejection']) * data['amount_intensity'] * data['intraday_return']
    
    # Relative Volume-Price Momentum
    data['price_momentum'] = data['close'].pct_change(5)
    data['volume_momentum'] = data['volume'] / data['volume_ma_5']
    data['efficiency_ratio'] = abs(data['intraday_return']) / data['range'].replace(0, np.nan)
    data['signal_volume_price'] = data['price_momentum'] * data['volume_momentum'] * data['efficiency_ratio'] * data['amount']
    
    # Combine all signals with equal weighting
    signals = [
        'low_to_high_transition', 'high_to_low_decay', 'thin_to_thick_efficiency',
        'spread_persistence', 'weekly_vs_daily', 'historical_reactivation',
        'micro_macro_volume', 'volume_scaling', 'momentum_inflection',
        'range_breakout_prob', 'boundary_attraction', 'volume_accumulation',
        'gap_absorption', 'price_impact', 'signal_intraday_reversal',
        'signal_range_momentum', 'signal_volatility_jump', 'signal_amount_rejection',
        'signal_volume_price'
    ]
    
    # Calculate final factor as weighted average of normalized signals
    factor = pd.Series(0, index=data.index)
    for signal in signals:
        if signal in data.columns:
            signal_data = data[signal].fillna(0)
            # Normalize by cross-sectional z-score each day
            normalized = signal_data.groupby(signal_data.index).transform(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0)
            factor += normalized
    
    return factor
