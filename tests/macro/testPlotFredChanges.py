import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from macro.computeFredChanges import detect_frequency, compute_changes

FRED_DATA_FOLDER = "./macro/fredData"

def test_plot_fred_changes(filename):
    """
    Loads a single FRED CSV, detects frequency,
    computes changes, and plots them before saving.
    This is purely for local testing/visual checks.
    """
    filepath = os.path.join(FRED_DATA_FOLDER, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    # Load data
    df = pd.read_csv(filepath)
    if 'date' not in df.columns:
        print(f"No 'date' column found in {filepath}")
        return
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.set_index('date', inplace=True)
    
    # Identify the value column (assume there's only one metric)
    metric_cols = [col for col in df.columns if col.lower() not in ("date", "datetime")]
    if not metric_cols:
        print(f"No metric column to compute changes for in {filepath}")
        return
    value_col = metric_cols[0]

    # Detect frequency
    freq_guess = detect_frequency(df.index)
    if not freq_guess:
        freq_guess = "unknown_frequency"
        print(f"Unable to detect frequency for {filename}. Defaulting to {freq_guess}.")

    # Compute changes
    df_changes = compute_changes(df, freq_guess, value_col)

    # For this test, just plot the new columns if any
    new_cols = set(df_changes.columns) - set(df.columns)
    if not new_cols:
        print(f"No changes computed for {filename}; skipping plot.")
        return
    
    # Plot each new column separately for a quick visual
    fig, axes = plt.subplots(nrows=len(new_cols), ncols=1, figsize=(8, 5 * len(new_cols)))
    if len(new_cols) == 1:
        axes = [axes]  # Make it iterable if there's only one
    
    for ax, col in zip(axes, sorted(new_cols)):
        df_changes[col].plot(ax=ax, title=f"{filename} - {col}")
        ax.set_ylabel("% Change")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Get all CSV files in the FRED data folder
    fred_files = [f for f in os.listdir(FRED_DATA_FOLDER) if f.endswith('.csv')]
    
    if not fred_files:
        print(f"No CSV files found in {FRED_DATA_FOLDER}")
    else:
        print(f"Found {len(fred_files)} FRED data files")
        for file in fred_files:
            print(f"\nProcessing {file}...")
            test_plot_fred_changes(file)
            plt.close('all')  # Close plots to free memory
