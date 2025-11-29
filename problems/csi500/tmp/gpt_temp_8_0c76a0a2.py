import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic price returns and volumes
    df['prev_close'] = df['close'].shift(1)
    df['return_2d'] = df['close'].pct_change(2)
    df['return_5d'] = df['close'].pct_change(5)
    df['volume_2d_avg'] = df['volume'].rolling(window=2).mean()
    df['volume_5d_avg'] = df['volume'].rolling(window=5).mean()
    
    # Calculate intraday price levels (approximations)
    df['hour10_high'] = df['high'].rolling(window=60).max()  # Approximate hour 10 high
    df['hour10_low'] = df['low'].rolling(window=60).min()    # Approximate hour 10 low
    df['hour11_close'] = df['close'].shift(-1)  # Using next period as approximation
    df['hour13_close'] = df['close'].shift(-3)  # Using future periods as approximations
    df['hour14_close'] = df['close'].shift(-4)
    df['hour15_high'] = df['high'].shift(-5)   # Using future periods
    df['hour15_low'] = df['low'].shift(-5)
    
    # Volume approximations for intraday periods
    df['first_30min_volume'] = df['volume'] * 0.1  # Approximation
    df['last_30min_volume'] = df['volume'] * 0.15  # Approximation
    df['morning_volume'] = df['volume'] * 0.4      # Approximation
    df['afternoon_volume'] = df['volume'] * 0.6    # Approximation
    df['morning_amount'] = df['amount'] * 0.4      # Approximation
    df['afternoon_amount'] = df['amount'] * 0.6    # Approximation
    
    # Auction Amplitude Asymmetry Analysis
    df['opening_auction_amplitude'] = (df['open'] - df['prev_close']) * df['first_30min_volume']
    df['closing_auction_amplitude'] = (df['close'] - df['hour14_close']) * df['last_30min_volume']
    df['net_auction_asymmetry'] = df['opening_auction_amplitude'] - df['closing_auction_amplitude']
    
    # Intraday Range Achievement
    df['morning_range_efficiency'] = (df['hour11_close'] - df['open']) / (df['hour10_high'] - df['hour10_low'] + 1e-8)
    df['afternoon_range_efficiency'] = (df['close'] - df['hour13_close']) / (df['hour15_high'] - df['hour15_low'] + 1e-8)
    df['range_efficiency_bias'] = df['morning_range_efficiency'] - df['afternoon_range_efficiency']
    
    # Gap-Auction Integration
    df['opening_gap_intensity'] = abs(df['open'] - df['prev_close']) * df['first_30min_volume']
    df['gap_auction_alignment'] = df['opening_gap_intensity'] * np.sign(df['net_auction_asymmetry'])
    df['auction_divergence'] = df['gap_auction_alignment'] - df['return_5d']
    
    # Volume-Regime Auction Framework
    df['opening_concentration'] = df['first_30min_volume'] / (df['morning_volume'] + 1e-8)
    df['closing_concentration'] = df['last_30min_volume'] / (df['afternoon_volume'] + 1e-8)
    df['volume_timing_asymmetry'] = df['opening_concentration'] - df['closing_concentration']
    
    # Auction-Price Efficiency
    df['morning_auction_efficiency'] = (df['hour11_close'] - df['open']) * df['morning_amount']
    df['afternoon_auction_efficiency'] = (df['close'] - df['hour13_close']) * df['afternoon_amount']
    df['efficiency_differential'] = df['morning_auction_efficiency'] - df['afternoon_auction_efficiency']
    
    # Volume Acceleration Metrics
    df['volume_rank'] = df['volume'].rolling(window=5).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    df['price_acceleration'] = df['return_2d'] - df['return_5d']
    df['volume_acceleration'] = (df['volume_2d_avg'] - df['volume_5d_avg']) / (df['volume_5d_avg'] + 1e-8)
    
    # Auction State Transition System
    df['trend_continuation'] = (df['return_5d'] * df['return_2d'] > 0).astype(int)
    df['trend_reversal'] = (df['return_5d'] * df['return_2d'] < 0).astype(int)
    df['auction_strength'] = abs(df['return_5d'])
    
    # Intraday Auction Persistence
    df['opening_auction_persistence'] = np.sign(df['open'] - df['prev_close']) * df['net_auction_asymmetry']
    df['closing_auction_persistence'] = np.sign(df['close'] - df['open']) * df['range_efficiency_bias']
    df['full_day_auction'] = df['opening_auction_persistence'] + df['closing_auction_persistence']
    
    # Regime-Sensitive Auction
    volume_median_3d = df['volume'].rolling(window=3).median()
    df['high_volume_auction'] = df['full_day_auction'] * (df['volume'] > volume_median_3d).astype(int)
    df['low_volume_auction'] = df['full_day_auction'] * (df['volume'] < volume_median_3d).astype(int)
    df['regime_weighted_auction'] = df['high_volume_auction'] + df['low_volume_auction']
    
    # Asymmetry-Efficiency Integration
    df['directional_efficiency'] = df['net_auction_asymmetry'] * df['efficiency_differential']
    df['timing_alignment'] = df['range_efficiency_bias'] * df['volume_timing_asymmetry']
    df['combined_auction_asymmetry'] = df['directional_efficiency'] + df['timing_alignment']
    
    # Acceleration-Enhanced Auction
    df['price_acceleration_component'] = df['combined_auction_asymmetry'] * df['price_acceleration']
    df['volume_acceleration_component'] = df['combined_auction_asymmetry'] * df['volume_acceleration']
    df['acceleration_weighted_auction'] = df['price_acceleration_component'] + df['volume_acceleration_component']
    
    # Regime-Dependent Amplification
    df['volume_regime_modulation'] = df['acceleration_weighted_auction'] * (1 - df['volume_rank'])
    df['gap_auction_enhancement'] = df['volume_regime_modulation'] * df['gap_auction_alignment']
    
    # True Range calculation
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    df['auction_divergence_adjustment'] = df['gap_auction_enhancement'] / (df['true_range'] + 1e-8)
    
    # Adaptive Alpha Signal Construction
    df['continuation_signal'] = df['auction_divergence_adjustment'] * df['trend_continuation']
    df['reversal_signal'] = df['auction_divergence_adjustment'] * df['trend_reversal']
    
    # Volume regime change detection
    df['volume_regime'] = (df['volume'] > df['volume'].rolling(window=3).median()).astype(int)
    df['volume_regime_change'] = (df['volume_regime'] != df['volume_regime'].shift(1)).astype(int)
    df['regime_transition_signal'] = (df['continuation_signal'] + df['reversal_signal']) * df['volume_regime_change']
    
    # Auction-Asymmetry Integration
    df['regime_aligned_auction'] = df['regime_transition_signal'] * df['regime_weighted_auction']
    df['persistence_enhancement'] = df['regime_aligned_auction'] * df['full_day_auction']
    df['strength_weighted_signal'] = df['persistence_enhancement'] * df['auction_strength']
    
    # Final Alpha Output
    df['volume_timing_confirmation'] = df['strength_weighted_signal'] * df['volume_timing_asymmetry']
    df['efficiency_validation'] = df['volume_timing_confirmation'] * df['efficiency_differential']
    df['final_alpha'] = df['efficiency_validation'] * np.sign(df['closing_auction_amplitude'])
    
    # Return the final alpha factor
    return df['final_alpha']
