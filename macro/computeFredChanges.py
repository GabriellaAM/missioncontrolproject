import os
import pandas as pd
import numpy as np
from dateutil import rrule

# Paths
FRED_DATA_FOLDER = "./macro/fredData"
FRED_CHANGES_FOLDER = "./macro/fredChanges"

# Make sure output folder exists
os.makedirs(FRED_CHANGES_FOLDER, exist_ok=True)

def detect_frequency(dates):
    """
    Naive frequency detection based on average gap in days.
    Returns 'weekly', 'monthly', or 'quarterly'.
    """
    if len(dates) < 2:
        return None

    # Convert to list of dates if Series
    dates_list = dates.tolist() if hasattr(dates, 'tolist') else dates
    
    # Calculate average difference in days
    deltas = [(dates_list[i] - dates_list[i - 1]).days for i in range(1, len(dates_list))]
    avg_gap = np.mean(deltas)

    # Basic heuristics for frequency detection:
    if avg_gap <= 10:
        return "weekly"
    elif avg_gap <= 40:
        return "monthly"
    elif avg_gap <= 130:
        return "quarterly"
    else:
        return None

def compute_changes(df, freq_guess, value_col):
    """
    Computes only MoM and YoY changes regardless of frequency.
    Returns a DataFrame with added columns: 'MoM' and 'YoY'.
    """
    # Sort by date just in case
    df = df.sort_index()

    # Calculate MoM and YoY for all frequencies
    df["MoM"] = df[value_col].pct_change(1) * 100  # 1-period change
    df["YoY"] = df[value_col].pct_change(12) * 100  # 12-period change

    return df

def process_fred_data_file(filepath):
    """
    Reads a single FRED CSV, detects frequency, computes changes,
    and writes to the ./macro/fredChanges/ folder.
    """
    try:
        # Load CSV
        df = pd.read_csv(filepath)
        # Expect 'date' column
        if 'date' not in df.columns:
            return

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)
        df.set_index("date", inplace=True)

        # The non-date columns are potential values
        # We'll assume there's only 1 metric column in this simple approach
        metric_cols = [col for col in df.columns if col.lower() not in ("date", "datetime")]
        if not metric_cols:
            return

        value_col = metric_cols[0]

        # Detect frequency
        freq_guess = detect_frequency(df.index)
        if not freq_guess:
            # If we can't detect frequency, skip or mark as unknown
            freq_guess = "unknown_frequency"

        # Compute changes
        df_changes = compute_changes(df, freq_guess, value_col)

        # Build output filename
        base_name = os.path.splitext(os.path.basename(filepath))[0]  # e.g. 'consumerPriceIndex'
        out_filename = f"{base_name}_changes_{freq_guess}.csv"  # e.g. 'consumerPriceIndex_changes_monthly.csv'
        out_path = os.path.join(FRED_CHANGES_FOLDER, out_filename)

        # Save
        df_changes.to_csv(out_path, index=True)
        print(f"Processed {filepath} -> {out_path}")

    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    # Process all CSV files in FRED_DATA_FOLDER
    files = [os.path.join(FRED_DATA_FOLDER, f) for f in os.listdir(FRED_DATA_FOLDER) if f.endswith(".csv")]
    for file_path in files:
        process_fred_data_file(file_path)

if __name__ == "__main__":
    main()
