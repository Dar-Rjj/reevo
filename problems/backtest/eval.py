import numpy as np
import pandas as pd
import os
import sys
import logging
sys.path.insert(0, "../../../")

from scipy.stats import spearmanr
import gpt
from utils.utils import get_heuristic_name


possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)


def solve(market_data):
    # 计算每只股票的因子值
    market_data['factor'] = market_data.groupby('stock_code').apply(lambda x: heuristics(x)).reset_index(level=0, drop=True)

    # 计算未来6日收益率
    market_data['future_return_6d'] = market_data.groupby('stock_code')['close'].shift(-6) / market_data['close'] - 1

    # 取所有日期
    start_date = pd.Timestamp('2021-01-01')
    end_date = pd.Timestamp('2024-01-01')
    all_dates = market_data.index.get_level_values('date').unique()
    all_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]
    ic_values = []

    for date in all_dates:
        daily = market_data.xs(date, level='date')
        factors = daily['factor']
        returns = daily['future_return_6d']
        mask = factors.notna() & returns.notna()
        if mask.sum() >= 10:
            ic, _ = spearmanr(factors[mask], returns[mask])
            if not np.isnan(ic):
                ic_values.append(ic)

    return np.mean(ic_values) if ic_values else 0
    

if __name__ == "__main__":
    print("[*] Running ...")
    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    assert mood in ['train', 'val']
    
    basepath = os.path.dirname(__file__)
    dataset_path = os.path.join(basepath, "all_stock_data.csv")

    market_data = pd.read_csv(dataset_path, parse_dates=['date'])
    market_data.set_index(['stock_code', 'date'], inplace=True)
    market_data.sort_index(inplace=True)

    mean_ic = solve(market_data)
    print("[*] Average:")
    print(abs(mean_ic))