# %%

import pandas as pd
import yfinance as yf

df = pd.read_parquet('/Users/valter.rebelo/MissionControl/data_parquet/crypto_data/coingecko/bitcoin/data.parquet')
cats = pd.read_parquet('/Users/valter.rebelo/MissionControl/data_parquet/asset_categories/data.parquet')
dom = pd.read_parquet('/Users/valter.rebelo/MissionControl/data_parquet/market_data/dominance/data.parquet')

# %%
df
# %%
cats
# %%

dt = yf.download('DX-Y.NYB', start='2025-01-01', end='2025-07-29')
dt
# %%
# Download gold and copper futures data
gold = yf.download('GC=F', start='2010-01-01', end='2025-07-29')
copper = yf.download('HG=F', start='2010-01-01', end='2025-07-29')

# Align indices (outer join to keep all dates from both)
combined = pd.DataFrame(index=gold.index.union(copper.index))
combined['gold_close'] = gold['Close']
combined['copper_close'] = copper['Close']

# Calculate ratio
combined['ratio'] = combined['gold_close'] / combined['copper_close']

# Optionally, drop rows where either price is missing
gold_copper_ratio = combined[['ratio']].copy()

print("DataFrame shape:", gold_copper_ratio.shape)
print("\nFirst few rows:")
print(gold_copper_ratio.head())

# Plot the ratio throughout time
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(gold_copper_ratio.index, gold_copper_ratio['ratio'], label='Gold/Copper Ratio')
plt.title('Gold/Copper Ratio Over Time')
plt.xlabel('Date')
plt.ylabel('Gold/Copper Ratio')
plt.grid(True)
plt.legend()
plt.show()

# %%
# Plot stablecoin dominance as area chart
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.fill_between(dom['timestamp'], dom.stablecoin_dominance, alpha=0.5)
plt.title('Stablecoin Market Dominance Over Time')
plt.xlabel('Date')
plt.ylabel('Stablecoin Dominance')
plt.grid(True)
plt.show()

# %%
rty_ym = pd.read_parquet('/Users/valter.rebelo/MissionControl/data_parquet/macro_data/rty_ym_ratio/data.parquet')
rty_ym.plot(x = 'timestamp', y = 'value')
# %%
oil = pd.read_parquet('/Users/valter.rebelo/MissionControl/data_parquet/macro_data/oil/data.parquet')
oil
# %%
rty_ym


# %%

ym = pd.read_parquet('/Users/valter.rebelo/MissionControl/data_parquet/macro_data/ym/data.parquet')
ym

# %%

yc = pd.read_parquet('/Users/valter.rebelo/MissionControl/data_parquet/macro_data/yieldCurveRegime/data.parquet')
yc


