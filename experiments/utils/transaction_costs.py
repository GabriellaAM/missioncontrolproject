import pandas as pd
import numpy as np


def apply_transaction_costs(data: pd.DataFrame, signal_col: str = 'signal', 
                          asset_name: str = 'bitcoin') -> pd.DataFrame:
    """
    Apply transaction costs when position changes occur.
    
    Args:
        data: DataFrame with trading signals
        signal_col: Column name containing trading signals
        asset_name: Name of the asset (affects transaction cost rate)
    
    Returns:
        DataFrame with transaction costs applied to strategy returns
    """
    df = data.copy()
    
    # Define transaction cost rates
    transaction_cost_rate = 0.001 if asset_name.lower() == 'bitcoin' else 0.005  # 0.1% vs 0.5%
    
    # Identify position changes (signal different from previous signal)
    df['position_change'] = (df[signal_col] != df[signal_col].shift(1)).astype(int)
    
    # Apply transaction costs only when position changes
    # Transaction cost reduces returns by the specified percentage
    df['transaction_cost'] = df['position_change'] * transaction_cost_rate
    
    return df


def adjust_strategy_returns_for_costs(strategy_returns: pd.Series, 
                                    transaction_costs: pd.Series) -> pd.Series:
    """
    Adjust strategy returns by subtracting transaction costs.
    
    Args:
        strategy_returns: Series of strategy returns
        transaction_costs: Series of transaction costs
    
    Returns:
        Adjusted strategy returns
    """
    # Subtract transaction costs from returns
    # Only apply costs when there's a position change (cost > 0)
    adjusted_returns = strategy_returns - transaction_costs
    
    return adjusted_returns