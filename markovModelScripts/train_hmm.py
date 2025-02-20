import numpy as np
import random
from datetime import datetime
import os
import json
import argparse
import warnings 

from .data_loader import load_asset_data
from .features import apply_initial_features, process_features, save_scaler
from .data_utils import split_data
from .hmm import train_hmm, save_hmm_model
from .trading_strategy import generate_hmm_states, transact_at_open
from .visualization import plot_hmm_results
from .config import MACRO_FILES, FEATURES_BTC, FEATURES_ALTS, RANDOM_SEED

warnings.filterwarnings("ignore")

# Set random seeds
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_latest_version(base_path, asset):
    """Get the latest version number for a given asset"""
    pattern = f"hmm_{asset}_v"
    existing_versions = []
    
    if os.path.exists(base_path):
        for file in os.listdir(base_path):
            if file.startswith(pattern) and file.endswith('.pkl'):
                try:
                    version = int(file.split('_v')[-1].replace('.pkl', ''))
                    existing_versions.append(version)
                except ValueError:
                    continue
    
    return max(existing_versions, default=0) + 1

def train_model(asset, retrain=False, min_date=None):
    """
    Train or retrain HMM model
    
    Parameters:
    -----------
    asset : str
        Asset to train model for
    retrain : bool
        If True, overwrites existing model without new version
    min_date : str
        Minimum date for training data (optional)
    """
    # Load and preprocess data
    print(f"😀 Processing {asset} 😀 \n ")

    data = load_asset_data(asset, MACRO_FILES)
    
    print(f"😀 Loaded {asset} data 😀 \n  ")

    data = apply_initial_features(data, asset)
    
    print(f"😀 Applied initial features to {asset} data 😀 \n  ")

    data.dropna(inplace=True)
    
    print(f"😀 Dropped NaN values from {asset} data 😀 \n ")

    # Split data
    data.reset_index(inplace=True)
    train, embargo, test = split_data(data, train_size=0.8, embargo_size=0.01)
    
    if asset == "bitcoin":
        train = train.loc[train['Date'] > '2016-01-01']

    # Process features
    train_processed, scaler = process_features(asset, train, is_training=True)
    test_processed, _ = process_features(asset, test, scaler=scaler, is_training=False)

    # Select HMM features
    features = FEATURES_BTC if asset == "bitcoin" else FEATURES_ALTS
    
    f_train = train_processed.set_index('Date')[features]
    f_test = test_processed.set_index('Date')[features]

    # Train and predict HMM
    hmm = train_hmm(f_train)

    # Create directory structure using absolute paths
    base_path_model = os.path.join(PROJECT_ROOT, "modelsDirectory", "models", "hmm")
    base_path_scaler = os.path.join(PROJECT_ROOT, "modelsDirectory", "scalers")
    base_path_metadata = os.path.join(PROJECT_ROOT, "modelsDirectory", "metadata")

    # Create directories if they don't exist
    os.makedirs(base_path_model, exist_ok=True)
    os.makedirs(base_path_scaler, exist_ok=True)
    os.makedirs(base_path_metadata, exist_ok=True)

    print(f"Saving models to: {base_path_model}")  # Debug print

    if retrain:
        # Simple overwrite of existing model
        model_filename = f"hmm_{asset}.pkl"
        scaler_filename = f"scaler_{asset}.pkl"
    else:
        # Get new version number and add timestamp
        version = get_latest_version(base_path_model, asset)
        timestamp = datetime.now().strftime("%Y%m%d")
        model_filename = f"hmm_{asset}_v{version}_{timestamp}.pkl"
        scaler_filename = f"scaler_{asset}_v{version}_{timestamp}.pkl"
        metadata_filename = f"metadata_{asset}_v{version}_{timestamp}.json"

    # Full paths for saving
    model_path = os.path.join(base_path_model, model_filename)
    scaler_path = os.path.join(base_path_scaler, scaler_filename)
    
    print(f"Attempting to save model to: {model_path}")  # Debug print

    # Save model and scaler
    save_hmm_model(hmm, model_path)
    save_scaler(scaler, scaler_path)

    if not retrain:
        # Calculate state Sharpe ratios for metadata
        state_sharpes = [
            hmm.means_[i][0] / np.sqrt(np.diag(hmm.covars_[i])[0]) if np.diag(hmm.covars_[i])[0] > 0 else -np.inf 
            for i in range(3)
        ]
        
        # Save metadata only for versioned models
        metadata = {
            'asset': asset,
            'version': version,
            'timestamp': timestamp,
            'train_start': train['Date'].min().strftime('%Y-%m-%d'),
            'train_end': train['Date'].max().strftime('%Y-%m-%d'),
            'features': features,
            'metrics': {
                'state_sharpes': state_sharpes,
                'transition_matrix': hmm.transmat_.tolist(),
                'mean_returns': hmm.means_.tolist(),
                'covariances': hmm.covars_.tolist()
            }
        }
        metadata_path = os.path.join(base_path_metadata, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)

    print(f"Saved model version {version if not retrain else 'retrain'} for {asset}")

    # Generate trading signals and plot results
    for mode in ['long-only', 'long-short']:
        print(f"\n😀 Processing {asset} in {mode} mode 😀 \n")
        hmm_state_series, p_of_states_fav_unfav = generate_hmm_states(hmm, test_processed, f_test, mode=mode)
        result_df = transact_at_open(test_processed, hmm_state_series, p_of_states_fav_unfav)
        print(f"😀 Latest position is: {result_df['state'].iloc[-1]} 😀 \n")
        plot_hmm_results(result_df, asset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action='store_true', help='Retrain existing model')
    parser.add_argument('--asset', type=str, nargs='+', required=True, 
                       choices=['bitcoin', 'ethereum', 'solana', 'chainlink'],
                       help='One or more assets to train models for')
    args = parser.parse_args()

    # Loop through all provided assets
    for asset in args.asset:
        print(f"\n😀 Processing {asset} 😀 \n")
        train_model(asset, retrain=args.retrain)