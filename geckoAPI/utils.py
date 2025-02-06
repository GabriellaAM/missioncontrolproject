from datetime import datetime
import pandas as pd

def datetime_to_unix(dt_str, dt_format="%Y-%m-%d"):
    """
    Convert a datetime string to a Unix timestamp.
    
    :param dt_str: The datetime string to convert.
    :param dt_format: The format of the datetime string.
    :return: Unix timestamp as an integer.
    """
    dt = datetime.strptime(dt_str, dt_format)
    return int(dt.timestamp())

def parse_gecko_ohlcv(data):
    """
    Parse a list of lists into a Pandas DataFrame, convert Unix timestamps to dates,
    and shift dates by one day.
    
    :param data: List of lists containing Unix timestamps and OHLC data.
    :return: DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close'].
    """
    # Convert the data into a DataFrame
    df = pd.DataFrame(data, columns=['Timestamp', 'open', 'high', 'low', 'close'])
    
    # Convert Unix timestamp to human-readable date and format it to YYYY-MM-DD HH:MM:SS
    df['date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.strftime('%Y-%m-%d')
    
    # Shift the date by one day
    df['date'] = pd.to_datetime(df['date']) - pd.Timedelta(days=1)
    
    # Drop the 'Timestamp' column as it's no longer needed
    df.drop('Timestamp', axis=1, inplace=True)
    
    # Reorder columns to have 'Date' first
    df = df[['date', 'open', 'high', 'low', 'close']]
    
    return df

def parse_gecko_prices(data):
    """
    Parse a list of lists into a Pandas DataFrame, convert Unix timestamps to dates,
    drop the last row, format the date to YYYY-MM-DD, and shift dates by one day.
    
    :param data: Dictionary containing lists of lists for 'prices', 'market_caps', and 'total_volumes'.
    :return: DataFrame with columns ['Date', 'Close', 'Market Cap', 'Total Volume'].
    """
    # Extract the lists from the data dictionary
    prices = data.get('prices', [])
    market_caps = data.get('market_caps', [])
    total_volumes = data.get('total_volumes', [])
    
    # Create DataFrames for each list
    df_prices = pd.DataFrame(prices, columns=['Timestamp', 'close'])
    df_market_caps = pd.DataFrame(market_caps, columns=['Timestamp', 'market_cap'])
    df_total_volumes = pd.DataFrame(total_volumes, columns=['Timestamp', 'total_volume'])
    
    # Merge the DataFrames on the 'Timestamp' column
    df = df_prices.merge(df_market_caps, on='Timestamp').merge(df_total_volumes, on='Timestamp')
    
    # Convert Unix timestamp to human-readable date and format it to YYYY-MM-DD HH:MM:SS
    df['date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.strftime('%Y-%m-%d')
    
    # Shift the date by one day
    df['date'] = pd.to_datetime(df['date']) - pd.Timedelta(days=1)
    
    # Drop the 'Timestamp' column as it's no longer needed
    df.drop('Timestamp', axis=1, inplace=True)
    
    # Drop the last row
    df = df[:-1]
    
    # Reorder columns to have 'Date' first
    df = df[['date', 'close', 'market_cap', 'total_volume']]
    
    return df


