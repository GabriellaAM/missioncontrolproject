# %%
"""
Model Registration Script

This script helps you review experimental runs and register the best performing models
with proper asset-strategy naming and classification.

Usage:
1. Run cells sequentially to review runs
2. Select the best run for each asset-strategy combination
3. Register with proper naming and tags
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from config import setup_mlflow

# %%
########################################################
# Setup and Configuration
########################################################

# Setup MLflow
setup_mlflow()
client = mlflow.MlflowClient()

# Strategy type classification (for backward compatibility)
STRATEGY_TYPES = {
    'bollinger_bands': 'mean_reversion',
    'rsi': 'mean_reversion', 
    'mean_reversion': 'mean_reversion',
    'sma_crossover': 'trend_following',
    'ema_crossover': 'trend_following',
    'macd': 'trend_following',
    'momentum': 'trend_following',
    'trend_following': 'trend_following'
}

print("✅ MLflow setup complete")
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# %%
########################################################
# Asset & Strategy Selection
########################################################

# Available assets and strategies (modify as needed)
AVAILABLE_ASSETS = ['bitcoin', 'ethereum', 'chainlink', 'solana']
AVAILABLE_STRATEGIES = ['sma_crossover', 'bollinger_bands', 'ema_crossover', 'rsi', 'macd']

# Configuration for this registration session
TARGET_ASSET = 'bitcoin'  # Change this
TARGET_STRATEGY = 'sma_crossover'  # Change this

print(f"🎯 Target: {TARGET_ASSET} + {TARGET_STRATEGY}")
print(f"📊 Strategy Type: {STRATEGY_TYPES.get(TARGET_STRATEGY, 'unknown')}")

# %%
########################################################
# Review Available Experiments
########################################################

# List all experiments
experiments = client.search_experiments()
print("Available Experiments:")
for exp in experiments:
    if exp.name != "Default":
        runs_count = len(mlflow.search_runs([exp.experiment_id]))
        print(f"  - {exp.name} (ID: {exp.experiment_id}, Runs: {runs_count})")

# Select experiment (usually matches strategy name)
target_experiment = TARGET_STRATEGY
print(f"\n🔍 Looking for experiment: {target_experiment}")

# %%
########################################################
# Advanced MLflow Querying Examples
########################################################

print("🔍 Advanced Query Examples:")
print("="*50)

# Example 1: Query by strategy type
print("1. Query by strategy type:")
try:
    trend_following_runs = mlflow.search_runs(
        filter_string="tags.strategy_type = 'trend_following'",
        order_by=["metrics.wf_sharpe_ratio DESC"],
        max_results=5
    )
    print(f"   Found {len(trend_following_runs)} trend-following strategies")
except:
    print("   No runs found with strategy_type tag")

# Example 2: Query by performance threshold
print("2. Query high-performing runs:")
try:
    high_perf_runs = mlflow.search_runs(
        filter_string="metrics.wf_sharpe_ratio > 1.0 AND metrics.wf_perm_sharpe_pvalue < 0.05",
        order_by=["metrics.wf_sharpe_ratio DESC"],
        max_results=5
    )
    print(f"   Found {len(high_perf_runs)} high-performing runs")
except:
    print("   No runs found matching performance criteria")

# Example 3: Cross-asset comparison
print("3. Cross-asset comparison:")
try:
    multi_asset_runs = mlflow.search_runs(
        filter_string="tags.strategy_type = 'mean_reversion'",
        order_by=["metrics.wf_sharpe_ratio DESC"],
        max_results=10
    )
    if len(multi_asset_runs) > 0:
        asset_performance = multi_asset_runs.groupby('tags.asset')['metrics.wf_sharpe_ratio'].max()
        print("   Best Sharpe by asset:")
        for asset, sharpe in asset_performance.items():
            print(f"     {asset}: {sharpe:.3f}")
except:
    print("   No data for cross-asset comparison")

print("\n" + "="*50)

# %%
########################################################
# Search and Filter Runs
########################################################

# Search for runs matching our target asset and strategy
try:
    # Basic search for target asset/strategy
    runs = mlflow.search_runs(
        experiment_names=[target_experiment],
        filter_string=f"tags.asset = '{TARGET_ASSET}'",
        order_by=["metrics.wf_sharpe_ratio DESC"],
        max_results=20
    )
    
    # Alternative: Query by strategy type instead of experiment
    if len(runs) == 0:
        print(f"🔄 No runs in {target_experiment}, trying strategy type search...")
        strategy_type = STRATEGY_TYPES.get(TARGET_STRATEGY, 'unknown')
        runs = mlflow.search_runs(
            filter_string=f"tags.asset = '{TARGET_ASSET}' AND tags.strategy_type = '{strategy_type}'",
            order_by=["metrics.wf_sharpe_ratio DESC"],
            max_results=20
        )
    
    if len(runs) == 0:
        print(f"❌ No runs found for {TARGET_ASSET} with {TARGET_STRATEGY}")
        print("Available combinations:")
        all_runs = mlflow.search_runs(max_results=100)
        if len(all_runs) > 0:
            combos = all_runs[['tags.asset', 'tags.strategy_type']].value_counts()
            print(combos.head(10))
    else:
        print(f"✅ Found {len(runs)} runs for {TARGET_ASSET} + {TARGET_STRATEGY}")
        
except Exception as e:
    print(f"❌ Error searching runs: {e}")
    runs = pd.DataFrame()

# %%
########################################################
# Performance Analysis & Comparison
########################################################

if len(runs) > 0:
    # Key performance metrics to review
    key_metrics = [
        'wf_sharpe_ratio', 'wf_profit_factor', 'wf_total_return', 
        'wf_max_drawdown', 'wf_win_rate', 
        'wf_perm_sharpe_pvalue', 'wf_perm_profit_factor_pvalue'
    ]
    
    # Create comparison dataframe
    comparison_df = runs[['run_id', 'start_time'] + [f'metrics.{m}' for m in key_metrics if f'metrics.{m}' in runs.columns]].copy()
    comparison_df.columns = ['run_id', 'start_time'] + [m.replace('metrics.', '') for m in comparison_df.columns[2:]]
    
    # Sort by Sharpe ratio
    if 'wf_sharpe_ratio' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('wf_sharpe_ratio', ascending=False)
    
    print("🏆 Top performing runs:")
    print("=" * 100)
    
    # Display top 5 runs
    display_df = comparison_df.head(10).round(4)
    print(display_df.to_string(index=False))
    
    # Show best run details
    if len(comparison_df) > 0:
        best_run_id = comparison_df.iloc[0]['run_id']
        best_run = client.get_run(best_run_id)
        
        print(f"\n🥇 Best Run Details:")
        print(f"Run ID: {best_run_id}")
        print(f"Start Time: {best_run.info.start_time}")
        print(f"Status: {best_run.info.status}")
        
        # Show key metrics
        print("\n📊 Key Metrics:")
        for metric in key_metrics:
            value = best_run.data.metrics.get(f'wf_{metric.replace("wf_", "")}', 'N/A')
            if value != 'N/A':
                print(f"  {metric}: {value:.4f}")
else:
    print("❌ No runs to analyze")

# %%
########################################################
# Model Registration
########################################################

if len(runs) > 0:
    # Get the run to register (by default, the best one)
    selected_run_id = comparison_df.iloc[0]['run_id']  # Change index to select different run
    
    print(f"🎯 Selected run for registration: {selected_run_id}")
    
    # Confirm the run has a model artifact
    run = client.get_run(selected_run_id)
    artifacts = client.list_artifacts(selected_run_id)
    
    has_model = any('model' in artifact.path for artifact in artifacts)
    
    if has_model:
        # Generate model name: asset_strategy format
        model_name = f"{TARGET_ASSET}_{TARGET_STRATEGY}"
        strategy_type = STRATEGY_TYPES.get(TARGET_STRATEGY, 'unknown')
        
        print(f"📝 Model name: {model_name}")
        print(f"🏷️  Strategy type: {strategy_type}")
        
        # Check if model already exists
        try:
            existing_versions = client.search_model_versions(f"name='{model_name}'")
            print(f"ℹ️  Found {len(existing_versions)} existing versions of this model")
        except:
            print(f"ℹ️  This will be the first version of {model_name}")
        
        # Register the model
        print(f"\n🚀 Registering model...")
        try:
            model_version = mlflow.register_model(
                model_uri=f"runs:/{selected_run_id}/model",
                name=model_name,
                tags={
                    "strategy_type": strategy_type,
                    "asset": TARGET_ASSET,
                    "strategy": TARGET_STRATEGY,
                    "registration_date": pd.Timestamp.now().isoformat()
                }
            )
            
            print(f"✅ Model registered successfully!")
            print(f"   Model: {model_name}")
            print(f"   Version: {model_version.version}")
            print(f"   Strategy Type: {strategy_type}")
            
        except Exception as e:
            print(f"❌ Registration failed: {e}")
            
    else:
        print(f"❌ No model artifact found in run {selected_run_id}")
        print("Available artifacts:")
        for artifact in artifacts:
            print(f"  - {artifact.path}")
else:
    print("❌ No runs available for registration")

# %%
########################################################
# Model Verification & Testing
########################################################

# Verify the registered model can be loaded
model_name = f"{TARGET_ASSET}_{TARGET_STRATEGY}"

try:
    print(f"🔍 Verifying registered model: {model_name}")
    
    # Load the latest version
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
    
    print("✅ Model loaded successfully!")
    print(f"   Model type: {type(model)}")
    
    # Get model version info
    latest_versions = client.get_latest_versions(model_name, stages=["None"])
    if latest_versions:
        version = latest_versions[0]
        print(f"   Version: {version.version}")
        print(f"   Stage: {version.current_stage}")
        print(f"   Creation time: {version.creation_timestamp}")
        
        # Show model tags
        if hasattr(version, 'tags') and version.tags:
            print(f"   Tags: {version.tags}")
    
    # Test prediction capability (if you have sample data)
    print(f"\n💡 To use this model:")
    print(f"   model = mlflow.pyfunc.load_model('models:/{model_name}/latest')")
    print(f"   predictions = model.predict(your_data)")
    
except Exception as e:
    print(f"❌ Model verification failed: {e}")

# %%
########################################################
# Summary and Next Steps
########################################################

print("\n" + "="*60)
print("📋 REGISTRATION SUMMARY")
print("="*60)

print(f"Target Asset: {TARGET_ASSET}")
print(f"Target Strategy: {TARGET_STRATEGY}")
print(f"Model Name: {TARGET_ASSET}_{TARGET_STRATEGY}")
print(f"Strategy Type: {STRATEGY_TYPES.get(TARGET_STRATEGY, 'unknown')}")

if len(runs) > 0:
    print(f"Runs Reviewed: {len(runs)}")
    print(f"Selected Run: {selected_run_id}")

print(f"\n💡 Next Steps:")
print(f"1. Repeat for other asset-strategy combinations")
print(f"2. Use the model: mlflow.pyfunc.load_model('models:{TARGET_ASSET}_{TARGET_STRATEGY}/latest')")
print(f"3. Consider promoting to 'Staging' or 'Production' stage when ready")

print(f"\n🎯 To register more models:")
print(f"   - Change TARGET_ASSET and TARGET_STRATEGY variables")
print(f"   - Re-run the cells")

# %%