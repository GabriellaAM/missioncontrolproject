#!/usr/bin/env python
"""Test script to verify automatic signature generation for rules-based strategies."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mlflow
import pandas as pd
from config import setup_mlflow

def test_strategy_signature_detection(strategy_class, strategy_name, asset='bitcoin'):
    """Test signature generation for a strategy."""
    print(f"\n🧪 Testing {strategy_name}...")
    
    # Create strategy instance
    strategy = strategy_class(asset=asset)
    
    # Test automatic feature detection
    try:
        input_example = strategy.get_input_example()
        print(f"✅ Generated input example:")
        print(f"   Columns: {list(input_example.columns)}")
        print(f"   Shape: {input_example.shape}")
        
        # Test the strategy can process this input
        try:
            params = getattr(strategy, 'default_params', {'fast_period': 10, 'slow_period': 30})
            result = strategy.calculate_signals(input_example, params)
            print(f"✅ Strategy can process generated input")
            print(f"   Output columns: {list(result.columns)}")
            print(f"   Has signal: {'signal' in result.columns}")
            
            return True
            
        except Exception as e:
            print(f"❌ Strategy failed to process input: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to generate input example: {e}")
        return False

def test_model_saving_with_signature(strategy_class, strategy_name, asset='bitcoin'):
    """Test complete model saving with signature."""
    print(f"\n🚀 Testing model saving for {strategy_name}...")
    
    # Setup MLflow
    setup_mlflow()
    
    experiment_name = "signature_test"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    mlflow.set_experiment(experiment_name)
    
    # Create strategy
    strategy = strategy_class(asset=asset)
    params = getattr(strategy, 'default_params', {'fast_period': 10, 'slow_period': 30})
    
    # Test model saving
    with mlflow.start_run(run_name=f"test_{strategy_name}") as run:
        # Log basic tags
        mlflow.set_tags({
            'asset': asset,
            'strategy_type': strategy.strategy_type,
            'strategy_basis': getattr(strategy, 'implementation_type', 'unknown')
        })
        
        try:
            model_uri = strategy.save_model(params)
            print(f"✅ Model saved: {model_uri}")
            
            # Try to load the model
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"✅ Model loaded successfully")
            
            # Check if it has a signature
            if hasattr(model, 'metadata') and hasattr(model.metadata, 'signature'):
                signature = model.metadata.signature
                if signature and signature.inputs:
                    print(f"✅ Model has signature with inputs: {signature.inputs}")
                else:
                    print(f"⚠️  Model loaded but signature is empty")
            else:
                print(f"⚠️  Model loaded but no signature found")
            
            return True
            
        except Exception as e:
            print(f"❌ Model saving/loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("🚀 Testing automatic signature generation...\n")
    
    # Test SMA Crossover
    from strategies.rules_based.sma_crossover import SMACrossoverStrategy
    success1 = test_strategy_signature_detection(SMACrossoverStrategy, "SMA Crossover")
    success2 = test_model_saving_with_signature(SMACrossoverStrategy, "sma_crossover")
    
    # Test Hilo Activator (should still work with existing method)
    from strategies.rules_based.hilo_activator import HiloActivatorStrategy
    success3 = test_strategy_signature_detection(HiloActivatorStrategy, "Hilo Activator") 
    success4 = test_model_saving_with_signature(HiloActivatorStrategy, "hilo_activator")
    
    print(f"\n🎯 SUMMARY:")
    print(f"SMA Crossover signature detection: {'✅' if success1 else '❌'}")
    print(f"SMA Crossover model saving: {'✅' if success2 else '❌'}")
    print(f"Hilo Activator signature detection: {'✅' if success3 else '❌'}")
    print(f"Hilo Activator model saving: {'✅' if success4 else '❌'}")
    
    if all([success1, success2, success3, success4]):
        print(f"\n🎉 All tests passed!")
    else:
        print(f"\n❌ Some tests failed")