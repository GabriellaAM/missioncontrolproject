#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to use the MarkovVolModel for predicting
volatility states with new Bitcoin data, especially for daily updates.

This shows two approaches:
1. Direct use of MarkovVolModel
2. Using TrendAnalyzer's interface to MarkovVolModel
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soros_system.analysis.markov_vol_model import MarkovVolModel
from soros_system.main import TrendAnalyzer

def example_direct_usage():
    """
    Example of direct usage of MarkovVolModel for predicting volatility.
    """
    print("\n=== Example 1: Direct usage of MarkovVolModel ===")
    
    # Define paths
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    btc_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'data', 'micro', 'candleData', 'bitcoin_candles.csv')
    
    # 1. Load an existing model or train a new one
    try:
        # Try to find most recent model
        model = MarkovVolModel(btc_data_path=btc_data_path)
        print("Loaded existing model or initialized a new one")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 2. Train model if needed
    if model.markov is None:
        print("Training new model...")
        success = model.train()
        if not success:
            print("Error training model")
            return
        print("Model trained successfully")
        
        # Save the model
        model_path = model.save_model()
        print(f"Model saved to {model_path}")
    
    # 3. Get volatility state for a known date
    known_date = "2023-01-01"
    state = model.get_volatility_state(known_date)
    print(f"Volatility state for {known_date}: {state}")
    
    # 4. Simulate new data point (tomorrow's close price)
    # Let's assume we have a new Bitcoin price for today
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Get the last known price from the data
    btc_data = pd.read_csv(btc_data_path)
    btc_data['date'] = pd.to_datetime(btc_data['date'])
    btc_data = btc_data.sort_values('date')
    last_price = btc_data['close'].iloc[-1]
    
    # Simulate a new price - 5% up from last price
    new_price = last_price * 1.05
    print(f"Last known price: {last_price}")
    print(f"Simulated new price: {new_price}")
    
    # 5. Predict volatility state for the new price
    state, signal, probability = model.predict_volatility_for_new_data(
        price_data=new_price,
        current_date=today
    )
    
    print(f"Predicted volatility for {today} with price {new_price}:")
    print(f"  - State: {state}")
    print(f"  - Signal: {signal} (1=low volatility, -1=high volatility)")
    print(f"  - Probability of high volatility: {probability:.4f}")
    
    # 6. Alternatively, provide log return directly
    log_return = 0.03  # Example: 3% positive log return
    state, signal, probability = model.predict_volatility_for_new_data(
        log_return=log_return,
        current_date=today
    )
    
    print(f"Predicted volatility for {today} with log return {log_return}:")
    print(f"  - State: {state}")
    print(f"  - Signal: {signal}")
    print(f"  - Probability of high volatility: {probability:.4f}")
    
    return model

def example_trend_analyzer_usage():
    """
    Example of using TrendAnalyzer interface to predict volatility.
    """
    print("\n=== Example 2: Using TrendAnalyzer Interface ===")
    
    # Define paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data', 'micro', 'candleData')
    btc_data_path = os.path.join(data_path, 'bitcoin_candles.csv')
    
    # 1. Initialize TrendAnalyzer with Bitcoin data
    analyzer = TrendAnalyzer(
        asset_ids=['bitcoin'],
        data_path=data_path,
        btc_data_path=btc_data_path
    )
    print("Initialized TrendAnalyzer")
    
    # 2. Simulate new BTC price data (5% up from last known price)
    btc_data = pd.read_csv(btc_data_path)
    btc_data['date'] = pd.to_datetime(btc_data['date'])
    btc_data = btc_data.sort_values('date')
    last_price = btc_data['close'].iloc[-1]
    new_price = last_price * 1.05
    
    print(f"Last known BTC price: {last_price}")
    print(f"Simulated new price: {new_price}")
    
    # 3. Predict volatility for the new price
    today = datetime.now().strftime("%Y-%m-%d")
    prediction = analyzer.predict_volatility_for_new_data(
        btc_price=new_price,
        prediction_date=today
    )
    
    print("Volatility prediction:")
    for key, value in prediction.items():
        print(f"  - {key}: {value}")
    
    # 4. Update model with a new data point
    # Create a DataFrame with a new data point
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    new_data = pd.DataFrame({
        'date': [tomorrow],
        'open': [new_price * 0.99],
        'high': [new_price * 1.02],
        'low': [new_price * 0.98],
        'close': [new_price * 1.01],  # 1% further increase
        'volume': [btc_data['volume'].mean()]  # Average volume
    })
    
    # Update the model (without retraining)
    print(f"Updating model with new data point for {tomorrow}")
    success = analyzer.update_volatility_model_with_new_data(new_data, retrain=False)
    
    if success:
        print("Model updated successfully")
        
        # Predict volatility for the next day after update
        day_after = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        next_price = new_price * 1.01 * 0.97  # 3% drop from tomorrow's price
        
        prediction = analyzer.predict_volatility_for_new_data(
            btc_price=next_price,
            prediction_date=day_after
        )
        
        print(f"Prediction for {day_after} with price {next_price}:")
        for key, value in prediction.items():
            print(f"  - {key}: {value}")
    else:
        print("Failed to update model")
    
    return analyzer

if __name__ == "__main__":
    # Run both examples
    model = example_direct_usage()
    analyzer = example_trend_analyzer_usage()
    
    print("\nDone!") 