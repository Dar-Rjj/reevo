import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Calculate basic price metrics
    df['prev_close'] = df['close'].shift(1)
    df['price_change'] = df['close'] - df['prev_close']
    df['return'] = df['price_change'] / df['prev_close']
    df['lag_return'] = df['return'].shift(1)
    df['midpoint'] = (df['high'] + df['low']) / 2
    
    # Volatility Regime Classification
    df['hl_range'] = (df['high'] - df['low']) / df['prev_close']
    df['short_vol'] = df['hl_range'].rolling(window=3, min_periods=2).std()
    df['medium_vol'] = df['hl_range'].rolling(window=10, min_periods=5).std()
    df['regime_ratio'] = df['short_vol'] / df['medium_vol']
    df['regime_ratio'] = df['regime_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Regime-Adaptive Breakout Signal
    df['rolling_high_5d'] = df['high'].rolling(window=5, min_periods=3).max()
    df['raw_breakout'] = (df['midpoint'] - df['rolling_high_5d']) / df['rolling_high_5d']
    df['vol_adjusted_breakout'] = df['raw_breakout'] / df['regime_ratio']
    df['breakout_signal'] = df['vol_adjusted_breakout'].rolling(window=3, min_periods=2).mean()
    
    # Volume Echo Efficiency
    df['volume_rank'] = df['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    high_volume_mask = df['volume_rank'] > 0.7
    df['price_vol_ratio'] = df['price_change'] / df['volume']
    df['volume_clustering'] = df['price_vol_ratio'].where(high_volume_mask, 0).rolling(window=10, min_periods=5).sum()
    
    vol_corr_window = 15
    df['vol_return_corr'] = df['volume'].rolling(window=vol_corr_window, min_periods=8).corr(df['return'])
    df['vol_lag_return_corr'] = df['volume'].rolling(window=vol_corr_window, min_periods=8).corr(df['lag_return'])
    df['volume_timing'] = df['vol_return_corr'] - df['vol_lag_return_corr']
    
    df['volume_efficiency'] = df['volume_clustering'] * df['volume_timing']
    
    # Session Momentum Divergence
    df['midday_price'] = (df['open'] + df['high'].rolling(window=2, min_periods=1).mean()) / 2
    df['morning_change'] = (df['midday_price'] - df['open']) / df['open']
    df['afternoon_change'] = (df['close'] - df['midday_price']) / df['midday_price']
    df['momentum_divergence'] = df['morning_change'] - df['afternoon_change']
    
    # Price-Level Rejection Analysis
    df['upper_rejection'] = (df['high'] - df['close']) / (df['high'] - df['low']).replace(0, np.nan)
    df['lower_rejection'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    df['max_rejection'] = df[['upper_rejection', 'lower_rejection']].max(axis=1)
    df['close_open_sign'] = np.sign(df['close'] - df['open'])
    df['efficiency_score'] = df['max_rejection'] * df['close_open_sign']
    
    # Generate Combined Alpha Factor
    df['base_signal'] = df['breakout_signal'] * df['volume_efficiency']
    df['momentum_enhanced'] = df['base_signal'] * df['momentum_divergence']
    df['alpha_factor'] = df['momentum_enhanced'] * df['efficiency_score']
    
    return df['alpha_factor']
