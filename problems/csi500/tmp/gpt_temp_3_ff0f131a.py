import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel cross-sectional alpha factors using price and volume data
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Factor 1: Intraday Compression Ratio with Volume Concentration
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['range_compression'] = (data['high'] - data['low']) / (data['prev_high'] - data['prev_low'])
    data['range_compression'] = data['range_compression'].replace([np.inf, -np.inf], np.nan)
    
    # Volume concentration in compressed ranges
    data['intraday_range'] = data['high'] - data['low']
    data['volume_density'] = data['volume'] / (data['intraday_range'] + 1e-8)
    compression_factor = data['range_compression'] * data['volume_density']
    
    # Factor 2: Transaction Momentum with Directional Consistency
    data['prev_amount'] = data['amount'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['current_vwap'] = data['amount'] / (data['volume'] + 1e-8)
    data['prev_vwap'] = data['prev_amount'] / (data['prev_volume'] + 1e-8)
    data['transaction_momentum'] = data['current_vwap'] / data['prev_vwap']
    
    # Directional consistency (3-day momentum alignment)
    data['price_change'] = data['close'].pct_change()
    data['vwap_change'] = data['current_vwap'].pct_change()
    directional_alignment = data['price_change'].rolling(window=3).corr(data['vwap_change'])
    transaction_factor = data['transaction_momentum'] * directional_alignment
    
    # Factor 3: Volatility Regime Quality
    data['daily_range_ratio'] = (data['high'] - data['low']) / (data['close'].shift(1) + 1e-8)
    data['range_sequence'] = data['daily_range_ratio'].rolling(window=5).std()
    
    # Price efficiency during volatility transitions
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    volatility_regime_factor = data['range_sequence'] * data['intraday_efficiency']
    
    # Factor 4: Pressure Gradient Alignment
    data['intraday_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['pressure_persistence'] = data['intraday_pressure'].rolling(window=3).mean()
    pressure_factor = data['intraday_pressure'] * data['pressure_persistence']
    
    # Factor 5: Liquidity Break Strength
    # Volume concentration at price barriers (using rolling high/low)
    data['rolling_high_5'] = data['high'].rolling(window=5).max()
    data['rolling_low_5'] = data['low'].rolling(window=5).min()
    data['at_high_barrier'] = (data['high'] >= data['rolling_high_5']).astype(int)
    data['at_low_barrier'] = (data['low'] <= data['rolling_low_5']).astype(int)
    
    data['barrier_volume'] = data['volume'] * (data['at_high_barrier'] + data['at_low_barrier'])
    data['amount_acceleration'] = data['amount'].pct_change(periods=2)
    liquidity_factor = data['barrier_volume'] * data['amount_acceleration']
    
    # Factor 6: Price-Volume Timing
    # Lead-lag correlation between price moves and volume spikes
    data['volume_spike'] = data['volume'] / data['volume'].rolling(window=10).mean()
    data['price_move'] = data['close'].pct_change()
    
    # 3-day rolling correlation between volume spikes and subsequent price moves
    price_volume_timing = []
    for i in range(len(data)):
        if i >= 3:
            window_data = data.iloc[i-3:i+1]
            if len(window_data) >= 3:
                corr = window_data['volume_spike'].iloc[:-1].corr(window_data['price_move'].iloc[1:])
                price_volume_timing.append(corr if not np.isnan(corr) else 0)
            else:
                price_volume_timing.append(0)
        else:
            price_volume_timing.append(0)
    
    data['price_volume_timing'] = price_volume_timing
    timing_factor = data['price_volume_timing']
    
    # Factor 7: Gap Microstructure
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_persistence'] = (data['close'] - data['open']) / (data['opening_gap'] + 1e-8)
    data['gap_persistence'] = data['gap_persistence'].replace([np.inf, -np.inf], np.nan)
    
    # Gap fill patterns with volume distribution
    data['gap_fill_ratio'] = abs(data['close'] - data['open'].shift(1)) / abs(data['opening_gap'] + 1e-8)
    gap_factor = data['gap_persistence'] * data['gap_fill_ratio']
    
    # Factor 8: Range Expansion Quality
    data['range_expansion'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)
    data['recent_volatility'] = data['close'].pct_change().rolling(window=10).std()
    
    # Intraday price path efficiency
    data['ideal_path'] = abs(data['close'] - data['open'])
    data['actual_path'] = abs(data['high'] - data['low'])
    data['path_efficiency'] = data['ideal_path'] / (data['actual_path'] + 1e-8)
    
    range_expansion_factor = data['range_expansion'] * data['path_efficiency'] / (data['recent_volatility'] + 1e-8)
    
    # Combine all factors with equal weighting
    factors = [
        compression_factor,
        transaction_factor,
        volatility_regime_factor,
        pressure_factor,
        liquidity_factor,
        timing_factor,
        gap_factor,
        range_expansion_factor
    ]
    
    # Normalize each factor and combine
    combined_factor = pd.Series(0, index=data.index)
    for factor in factors:
        if isinstance(factor, pd.Series):
            normalized_factor = (factor - factor.mean()) / (factor.std() + 1e-8)
            combined_factor += normalized_factor
    
    # Final normalization
    final_factor = (combined_factor - combined_factor.mean()) / (combined_factor.std() + 1e-8)
    
    return final_factor
