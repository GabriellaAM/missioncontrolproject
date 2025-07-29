# %%

import pandas as pd
import yfinance as yf

df = pd.read_parquet('/Users/valter.rebelo/MissionControl/data_parquet/crypto_data/arbitrum/data.parquet')
cats = pd.read_parquet('/Users/valter.rebelo/MissionControl/data_parquet/asset_categories/data.parquet')


# %%
cats
# %%

dt = yf.download('DX-Y.NYB', start='2025-01-01', end='2025-07-29')
dt
# %%
dt = yf.download('^MOVE', start='2023-01-01', end='2025-07-29')
dt
# %%
