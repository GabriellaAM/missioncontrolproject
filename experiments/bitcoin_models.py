# %%
"""
Bitcoin Models Experiments - SOROS Lab
"""

from config import setup_mlflow, create_experiment
import mlflow
import sys
import os
import pandas as pd


# %%

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup MLflow
setup_mlflow()
create_experiment("bitcoin-models")
mlflow.set_experiment("bitcoin-models")

# %%
# Load features using the new FeatureLoader system

from utils.feature_loader import FeatureLoader

# Initialize feature loader with global date range
loader = FeatureLoader(start_date='2020-01-01', end_date='2023-01-01')

# Build feature set with Bitcoin + macro features
features_df = loader.build_feature_set(
    crypto_assets=['ethereum', 'bitcoin', 'solana'],
    fred_indicators={'creditSpreads': 'credit_spread'},
    #yahoo_tickers={'vix': 'vix'},
    calculated_features={'rty_ym_ratio': 'rty_ym_ratio'}
)

print(f"Loaded {len(features_df)} rows of combined data")
print("\nColumns:", features_df.columns.tolist())
print("\nSample data:")
print(features_df.head(20))


# %%
# Start MLflow Run with Feature Tracking
with mlflow.start_run(run_name="bitcoin-ethereum-solana-macro-experiment"):
    
    # Log Feature Metadata and Statistics to MLflow
    loader.log_all_to_mlflow(features_df)
    
    # Log Basic Parameters
    mlflow.log_param("model_type", "placeholder_model")
    mlflow.log_param("assets", "bitcoin, ethereum, solana")
    mlflow.log_param("macro_features", "credit_spread,vix,rty_ym_ratio")
    
    # %%
    # Feature Engineering (add your custom features here)
    # features_df['bitcoin_returns'] = features_df['bitcoin_close'].pct_change()
    # features_df['ethereum_btc_ratio'] = features_df['ethereum_close_btc']
    # Add more features as needed
    
    # %%
    # Model Training Section
    # X = features_df.drop(['timestamp', 'target_column'], axis=1)
    # y = features_df['target_column']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 
    # model = YourModel()
    # model.fit(X_train, y_train)
    
    # %%
    # Model Testing & Predictions
    # y_pred = model.predict(X_test)
    
    # %%
    # Calculate & Log Metrics
    # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # r2 = r2_score(y_test, y_pred)
    # mae = mean_absolute_error(y_test, y_pred)
    # 
    # mlflow.log_metric("rmse", rmse)
    # mlflow.log_metric("r2_score", r2)
    # mlflow.log_metric("mae", mae)
    
    # %%
    # Log Model (when you have one)
    # mlflow.sklearn.log_model(model, "model")
    # mlflow.pytorch.log_model(model, "model")
    
    print("✅ Experiment with complete feature tracking logged to MLflow")
    print("🔍 Check MLflow UI to see feature metadata, statistics, and data quality reports")