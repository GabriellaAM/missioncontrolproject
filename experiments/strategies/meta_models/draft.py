#%%
import pandas as pd 
import numpy as np
import os
import sys

# Add parent directory to sys.path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.feature_loader import FeatureLoader

asset = ['bitcoin']

feats = FeatureLoader(start_date='2020-01-01', end_date='2024-12-31')

feats = feats.build_feature_set(
    crypto_assets=asset[0],
    fred_indicators=['creditSpreads', 'treasury5YInflationExpectation'],
    yahoo_tickers=['vix', 'move'],
    calculated_features={'yieldCurveRegime': ['regime', 'spread_2s10s'],
                         'rty_ym_ratio': 'value'}
)
feats.rename(columns={'value': 'rty_ym_ratio'}, inplace=True)
feats


# %% 

import pandas as pd 

df = pd.read_parquet('/Users/valter.rebelo/MissionControl/data_parquet/macro_data/calculated/rty_ym_ratio')

df.head()

