import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Regime Adjusted Opening Range Efficiency
    # Calculate Multi-Timeframe Opening Range Breakout
    data['short_up_breakout'] = (data['high'] - data['open']) / data['open']
    data['short_down_breakout'] = (data['open'] - data['low']) / data['open']
    
    # Compute Medium-Term Opening Range
    data['opening_range_magnitude'] = (data['high'] - data['low']) / data['open']
    data['avg_5d_opening_range'] = data['opening_range_magnitude'].rolling(window=5, min_periods=3).mean()
    data['range_efficiency_ratio'] = data['short_up_breakout'] / (data['avg_5d_opening_range'] + 1e-8)
    
    # Assess Volatility Compression State
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['avg_10d_range'] = data['daily_range'].rolling(window=10, min_periods=7).mean()
    data['avg_20d_range'] = data['daily_range'].rolling(window=20, min_periods=15).mean()
    data['vol_compression'] = data['avg_10d_range'] / (data['avg_20d_range'] + 1e-8)
    
    # Combine Range Efficiency with Volatility State
    data['compression_factor'] = 1.0 / (data['vol_compression'] + 1e-8)
    data['base_signal'] = data['range_efficiency_ratio'] * data['compression_factor']
    
    # Apply Volume-Weighted Confirmation
    data['avg_5d_volume'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_acceleration'] = data['volume'] / (data['avg_5d_volume'] + 1e-8)
    data['volume_trend'] = data['volume_acceleration'].rolling(window=3, min_periods=2).mean()
    
    data['factor_1'] = data['base_signal'] * data['volume_trend']
    
    # Amount-Concentrated Multi-Timeframe Reversal
    # Calculate Price-Level Dependent Reversal
    data['high_252d'] = data['high'].rolling(window=252, min_periods=200).max()
    data['low_252d'] = data['low'].rolling(window=252, min_periods=200).min()
    data['upside_potential'] = data['close'] / (data['high_252d'] + 1e-8)
    data['downside_risk'] = data['close'] / (data['low_252d'] + 1e-8)
    
    # Compute Multi-Timeframe Reversal Signal
    data['ret_2d'] = data['close'].pct_change(periods=2)
    data['ret_5d'] = data['close'].pct_change(periods=5)
    data['reversal_signal'] = data['ret_2d'] - data['ret_5d']
    
    # Evaluate Amount Concentration Signal
    data['amount_concentration'] = data['amount'] / ((data['high'] - data['low']) + 1e-8)
    data['amount_trend'] = data['amount_concentration'].rolling(window=3, min_periods=2).mean()
    data['amount_autocorr'] = data['amount_concentration'].rolling(window=5, min_periods=3).apply(
        lambda x: x.autocorr(lag=1) if len(x) >= 3 else np.nan, raw=False
    )
    
    # Combine Reversal with Amount Confirmation
    data['reversal_amount'] = data['reversal_signal'] * data['amount_concentration']
    
    # Apply Dynamic Volatility Filtering
    data['vol_persistence'] = data['daily_range'].rolling(window=5, min_periods=3).std()
    data['trading_activity'] = data['volume'].rolling(window=10, min_periods=7).apply(
        lambda x: (x > 0).sum(), raw=False
    )
    
    data['factor_2'] = data['reversal_amount'] * data['vol_persistence'] * data['trading_activity']
    
    # Intraday Sentiment Convergence with Range Expansion
    # Calculate Multi-Session Sentiment Bias
    data['morning_sentiment_up'] = (data['high'] - data['open']) / (data['open'] + 1e-8)
    data['morning_sentiment_down'] = (data['open'] - data['low']) / (data['open'] + 1e-8)
    
    # For simplicity, using midpoint as proxy for afternoon session
    data['midpoint'] = (data['high'] + data['low']) / 2
    data['afternoon_sentiment_up'] = (data['high'] - data['midpoint']) / (data['midpoint'] + 1e-8)
    data['afternoon_sentiment_down'] = (data['midpoint'] - data['low']) / (data['midpoint'] + 1e-8)
    
    data['sentiment_consistency'] = (data['morning_sentiment_up'] + data['afternoon_sentiment_up']) - \
                                   (data['morning_sentiment_down'] + data['afternoon_sentiment_down'])
    
    # Assess Range Expansion Momentum
    data['prev_range'] = data['daily_range'].shift(1)
    data['range_expansion'] = np.log(data['daily_range'] / (data['prev_range'] + 1e-8))
    data['directional_efficiency'] = (data['close'] - data['open']) / ((data['high'] - data['low']) + 1e-8)
    
    # Combine Sentiment Convergence with Expansion
    data['sentiment_expansion'] = data['sentiment_consistency'] * data['range_expansion']
    
    # Apply Volume Distribution Validation
    data['volume_ratio'] = data['volume'] / (data['volume'].rolling(window=5, min_periods=3).mean() + 1e-8)
    
    data['factor_3'] = data['sentiment_expansion'] * data['volume_ratio']
    
    # Volatility-Adjusted Momentum Divergence Efficiency
    # Calculate Multi-Timeframe Momentum
    data['intraday_strength'] = (data['high'] - data['low']) / (data['close'] + 1e-8)
    data['momentum_3d'] = data['intraday_strength'].rolling(window=3, min_periods=2).mean()
    
    # Overnight vs Intraday Divergence
    data['overnight_return'] = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    data['overnight_persistence'] = data['overnight_return'].rolling(window=3, min_periods=2).mean()
    data['momentum_divergence'] = data['intraday_strength'] - data['overnight_persistence']
    
    # Evaluate Volatility Regime Context
    data['vol_regime'] = data['avg_10d_range'] / (data['avg_20d_range'] + 1e-8)
    data['vol_adjustment'] = np.where(data['vol_regime'] > 1, 0.7, 1.3)
    
    # Combine Momentum with Volatility Adjustment
    data['adjusted_momentum'] = data['momentum_divergence'] * data['vol_adjustment']
    
    # Apply Opening Range Confirmation
    data['first_hour_range'] = (data['high'] - data['low']) / data['open']
    data['opening_efficiency'] = data['first_hour_range'] / (data['daily_range'] + 1e-8)
    
    data['factor_4'] = data['adjusted_momentum'] * data['opening_efficiency']
    
    # Combine all factors with equal weights
    factors = [data['factor_1'], data['factor_2'], data['factor_3'], data['factor_4']]
    
    # Normalize each factor and combine
    combined_factor = pd.Series(0, index=data.index)
    for factor in factors:
        normalized_factor = (factor - factor.rolling(window=252, min_periods=200).mean()) / \
                           (factor.rolling(window=252, min_periods=200).std() + 1e-8)
        combined_factor = combined_factor + normalized_factor
    
    # Final normalization
    final_factor = (combined_factor - combined_factor.rolling(window=252, min_periods=200).mean()) / \
                   (combined_factor.rolling(window=252, min_periods=200).std() + 1e-8)
    
    return final_factor
