import pandas as pd

def load_macro_data(asset_data, macro_files, resample_freq='D'):

    merged_data = asset_data.copy()

    for file in macro_files:

        macro_df = pd.read_csv(f"data/macro/fredData/{file}.csv")
        macro_df['date'] = pd.to_datetime(macro_df['date'])
        macro_df.set_index('date', inplace=True)

        if resample_freq == 'D':
            macro_df = macro_df.resample('D').ffill(limit=30).bfill()

        merged_data = pd.merge(merged_data, macro_df, left_index=True, right_index=True, how='left')
        null_mask = merged_data[macro_df.columns].isnull()

        if null_mask.any().any():
            print(f"\nGaps found in {file}:")

            for column in macro_df.columns:

                if null_mask[column].any():
                    gap_dates = merged_data[null_mask[column]].index
                    print(f"\n{column} gaps at dates:")

                    for date in gap_dates:
                        print(f"  {date.strftime('%Y-%m-%d')}")

            merged_data[macro_df.columns] = merged_data[macro_df.columns].ffill().bfill()

    return merged_data

def load_asset_data(asset, macro_files):

    df_1 = pd.read_csv(f"data/micro/candleData/{asset}_candles.csv")

    df_1['date'] = pd.to_datetime(df_1['date'])

    df_1.set_index('date', inplace=True)

    df_2 = pd.read_csv(f"data/micro/assetData/{asset}.csv")

    df_2['date'] = pd.to_datetime(df_2['date'])

    df_2.set_index('date', inplace=True)

    data = pd.merge(df_1, df_2[['total_volume', 'market_cap']], left_index=True, right_index=True, how='left')

    data = load_macro_data(data, macro_files)
    
    data.rename(columns={
        'total_volume': 'Volume', 'open': 'Open', 'high': 'High',
        'low': 'Low', 'close': 'Close', 'market_cap': 'marketCap'
    }, inplace=True)

    data.dropna(inplace=True)

    data.index.name = 'Date'
    
    return data