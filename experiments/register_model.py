#!/usr/bin/env python

import argparse
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mlflow
import mlflow.sklearn
from config import setup_mlflow
import importlib.util


def get_run_info(run_id):
    """Get detailed information about an MLflow run."""
    try:
        client = mlflow.MlflowClient()
        run = client.get_run(run_id)
        artifacts = client.list_artifacts(run_id)
        
        # Check for model by trying to load it (more reliable than artifact check)
        has_model = False
        try:
            mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
            has_model = True
        except:
            # Fallback to artifact path check
            has_model = any(artifact.path == 'model' for artifact in artifacts)
        
        has_strategy_config = any('strategy_config' in artifact.path for artifact in artifacts)
        
        return {
            'run': run,
            'artifacts': artifacts,
            'has_model': has_model,
            'has_strategy_config': has_strategy_config
        }
    except Exception as e:
        print(f"❌ Error retrieving run {run_id}: {e}")
        return None


def display_run_summary(run_info):
    """Display a summary of the run before registration."""
    run = run_info['run']
    
    print(f"\n📋 RUN SUMMARY")
    print(f"{'='*50}")
    print(f"Run ID: {run.info.run_id}")
    print(f"Status: {run.info.status}")
    print(f"Start Time: {datetime.fromtimestamp(run.info.start_time/1000)}")
    
    # Display key tags
    tags = run.data.tags
    asset = tags.get('asset', 'unknown')
    strategy_type = tags.get('strategy_type', 'unknown')  # Algorithmic approach
    strategy_basis = tags.get('strategy_basis', 'unknown')  # Implementation approach
    
    print(f"Asset: {asset}")
    print(f"Strategy Type: {strategy_type}")
    print(f"Strategy Basis: {strategy_basis}")
    
    # Display key metrics
    metrics = run.data.metrics
    key_metrics = [
        'wf_sharpe_ratio', 'wf_profit_factor', 'wf_total_return', 
        'wf_max_drawdown', 'wf_win_rate'
    ]
    
    print(f"\n📊 Key Metrics:")
    for metric in key_metrics:
        if metric in metrics:
            value = metrics[metric]
            if 'return' in metric or 'drawdown' in metric:
                print(f"  {metric}: {value:.1%}")
            else:
                print(f"  {metric}: {value:.4f}")
    
    if run_info['has_model']:
        print(f"\n📁 Model Artifact: ✅ Found")
        if run_info['has_strategy_config']:
            print(f"📁 Strategy Config: ✅ Found")
    else:
        print(f"\n📁 Model Artifact: ❌ Not found")
    
    return asset, strategy_type, strategy_basis


def register_model(run_id, model_name=None):
    """Register a model from an MLflow run."""
    
    # Setup MLflow
    setup_mlflow()
    client = mlflow.MlflowClient()
    
    # Get run information
    print(f"🔍 Retrieving run information...")
    run_info = get_run_info(run_id)
    
    if not run_info:
        return False
    
    if not run_info['has_model']:
        print(f"❌ No model artifact found in run {run_id}")
        print("Available artifacts:")
        for artifact in run_info['artifacts']:
            print(f"  - {artifact.path}")
        print(f"\nRun needs a 'model' artifact. Use strategy.save_model() during training.")
        return False
    
    # Display run summary
    asset, strategy_type, strategy_basis = display_run_summary(run_info)
    
    # Generate model name if not provided
    if not model_name:
        tags = run_info['run'].data.tags
        strategy = None
        
        if 'mlflow.runName' in tags:
            run_name_parts = tags['mlflow.runName'].split('_')
            if len(run_name_parts) >= 2:
                strategy = run_name_parts[0]
        
        if not strategy:
            experiment_id = run_info['run'].info.experiment_id
            experiment = client.get_experiment(experiment_id)
            strategy = experiment.name
        
        model_name = f"{asset}_{strategy}"
    
    print(f"\n🏷️  Model Name: {model_name}")
    
    # Check for existing versions
    try:
        existing_versions = client.search_model_versions(f"name='{model_name}'")
        if existing_versions:
            latest_version = max([int(v.version) for v in existing_versions])
            print(f"ℹ️  Next version will be: {latest_version + 1}")
        else:
            print(f"ℹ️  This will be the first version of {model_name}")
    except:
        print(f"ℹ️  This will be the first version of {model_name}")
    
    return model_name, run_info, asset, strategy_type, strategy_basis


def confirm_registration(model_name, run_id):
    """Ask user to confirm registration."""
    print(f"\n🚀 Ready to register model '{model_name}' from run {run_id}")
    response = input("Continue? [y/N]: ").strip().lower()
    return response in ['y', 'yes']


def do_registration(run_id, model_name, asset, strategy_type, strategy_basis, run_info):
    """Perform the actual model registration."""
    try:
        print(f"🚀 Registering model...")
        
        model_uri = f"runs:/{run_id}/model"
        print(f"📦 Registering model from: {model_uri}")
        
        # Register the model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags={
                "asset": asset,
                "strategy_type": strategy_type,
                "strategy_basis": strategy_basis,
                "registration_date": datetime.now().isoformat()
            }
        )
        
        print(f"✅ Model registered successfully!")
        print(f"   Model: {model_name}")
        print(f"   Version: {model_version.version}")
        print(f"   Asset: {asset}")
        print(f"   Strategy Type: {strategy_type}")
        print(f"   Strategy Basis: {strategy_basis}")
        
        print(f"\n💡 To use this model:")
        print(f"   model = mlflow.pyfunc.load_model('models:/{model_name}/latest')")
        
        return True
        
    except Exception as e:
        print(f"❌ Registration failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Register MLflow model from run ID')
    parser.add_argument('--run-id', required=True, help='MLflow run ID')
    parser.add_argument('--model-name', help='Custom model name (default: auto-generated from asset_strategy)')
    parser.add_argument('--yes', action='store_true', help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    print(f"🎯 Registering model from run: {args.run_id}")
    
    # Get run info and generate model name
    result = register_model(args.run_id, args.model_name)
    
    if not result:
        sys.exit(1)
    
    model_name, run_info, asset, strategy_type, strategy_basis = result
    
    # Confirm registration unless --yes flag is used
    if not args.yes:
        if not confirm_registration(model_name, args.run_id):
            print("❌ Registration cancelled")
            sys.exit(0)
    
    # Perform registration
    success = do_registration(args.run_id, model_name, asset, strategy_type, strategy_basis, run_info)
    
    if success:
        print(f"\n🎉 Registration complete!")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()