"""
Evaluation Metrics for Trading Strategies
"""

import numpy as np
from typing import Union, Optional, Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def calculate_sharpe_ratio(returns: np.ndarray, 
                          risk_free_rate: float = 0,
                          periods_per_year: int = 365) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate 
        periods_per_year: Number of periods in a year (365 for daily)
    
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)


def calculate_sortino_ratio(returns: np.ndarray,
                           risk_free_rate: float = 0,
                           periods_per_year: int = 365) -> float:
    """
    Calculate annualized Sortino ratio (uses downside deviation).
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate (default 2%)
        periods_per_year: Number of periods in a year (365 for daily)
    
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_deviation == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_deviation


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from returns.
    
    Args:
        returns: Array of returns
    
    Returns:
        Maximum drawdown (as negative percentage, e.g., -0.25 for 25% drawdown)
    """
    if len(returns) == 0:
        return 0.0
    
    cumulative_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    
    return np.min(drawdown)


def calculate_calmar_ratio(returns: np.ndarray,
                          periods_per_year: int = 365) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Args:
        returns: Array of returns
        periods_per_year: Number of periods in a year (365 for daily)
    
    Returns:
        Calmar ratio
    """
    max_dd = calculate_max_drawdown(returns)
    
    if max_dd == 0:
        return np.inf if np.mean(returns) > 0 else 0.0
    
    annualized_return = np.mean(returns) * periods_per_year
    return -annualized_return / max_dd  # Negative because max_dd is negative


def calculate_annualized_volatility(returns: np.ndarray,
                                   periods_per_year: int = 365) -> float:
    """
    Calculate annualized volatility.
    
    Args:
        returns: Array of returns
        periods_per_year: Number of periods in a year (365 for daily)
    
    Returns:
        Annualized volatility
    """
    if len(returns) == 0:
        return 0.0
    
    return np.std(returns) * np.sqrt(periods_per_year)


def calculate_profit_factor(returns: np.ndarray, 
                           signals: Optional[np.ndarray] = None) -> float:
    """
    Calculate profit factor for a trading strategy.
    
    Args:
        returns: Array of returns
        signals: Array of trading signals (+1 for long, -1 for short, 0 for no position)
                If None, assumes all returns are strategy returns
    
    Returns:
        Profit factor (gross profit / gross loss)
    """
    if signals is not None:
        strategy_returns = returns * signals
    else:
        strategy_returns = returns
    
    gross_profit = np.sum(strategy_returns[strategy_returns > 0])
    gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_information_ratio(strategy_returns: np.ndarray, 
                               benchmark_returns: np.ndarray,
                               periods_per_year: int = 365) -> float:
    """
    Calculate Information Ratio (excess return / tracking error).
    
    Args:
        strategy_returns: Array of strategy returns
        benchmark_returns: Array of benchmark returns (underlying asset)
        periods_per_year: Number of periods in a year (365 for daily)
    
    Returns:
        Annualized Information Ratio
    """
    if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    
    if len(strategy_returns) != len(benchmark_returns):
        raise ValueError("Strategy and benchmark returns must have the same length")
    
    # Calculate excess returns
    excess_returns = strategy_returns - benchmark_returns
    
    # Calculate tracking error (standard deviation of excess returns)
    tracking_error = np.std(excess_returns)
    
    if tracking_error == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    # Annualize both excess return and tracking error
    annualized_excess_return = np.mean(excess_returns) * periods_per_year
    annualized_tracking_error = tracking_error * np.sqrt(periods_per_year)
    
    return annualized_excess_return / annualized_tracking_error


def calculate_avg_time_underwater(returns: np.ndarray) -> float:
    """
    Calculate average time underwater (periods below peak).
    
    Args:
        returns: Array of returns
    
    Returns:
        Average number of periods underwater
    """
    if len(returns) == 0:
        return 0.0
    
    cumulative_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative_returns)
    
    # Identify underwater periods
    underwater = cumulative_returns < running_max
    
    # Count consecutive underwater periods
    underwater_periods = []
    current_period = 0
    
    for is_underwater in underwater:
        if is_underwater:
            current_period += 1
        else:
            if current_period > 0:
                underwater_periods.append(current_period)
                current_period = 0
    
    # Add last period if still underwater
    if current_period > 0:
        underwater_periods.append(current_period)
    
    if len(underwater_periods) == 0:
        return 0.0
    
    return np.mean(underwater_periods)


def extract_positions(signals: np.ndarray, returns: np.ndarray) -> List[Dict]:
    """
    Extract individual trading positions from signals and calculate their returns.
    
    Args:
        signals: Array of trading signals (1 for long, 0 for flat/cash)
        returns: Array of period returns
        
    Returns:
        List of position dictionaries with start, end, and total_return
    """
    if len(signals) != len(returns):
        raise ValueError("Signals and returns must have the same length")
    
    positions = []
    current_position = None
    
    for i in range(len(signals)):
        if signals[i] == 1 and (i == 0 or signals[i-1] == 0):
            # Position start - entering long position
            current_position = {'start': i}
            
        elif signals[i] == 0 and current_position is not None:
            # Position end - exiting position
            current_position['end'] = i - 1
            # Calculate cumulative return during position
            position_returns = returns[current_position['start']:i]
            current_position['total_return'] = np.sum(position_returns)
            current_position['periods'] = len(position_returns)
            positions.append(current_position)
            current_position = None
    
    # Handle case where strategy ends while in position
    if current_position is not None:
        current_position['end'] = len(signals) - 1
        position_returns = returns[current_position['start']:]
        current_position['total_return'] = np.sum(position_returns)
        current_position['periods'] = len(position_returns)
        positions.append(current_position)
    
    return positions


def create_confusion_matrix_plot(y_true: Union[List[int], np.ndarray],
                                y_pred: Union[List[int], np.ndarray],
                                title: str = "Position Profitability Confusion Matrix",
                                save_path: Optional[str] = None,
                                filename: Optional[str] = None,
                                log_to_mlflow: bool = False,
                                binary_conversion: str = 'none',
                                labels: Optional[List[int]] = None,
                                label_names: Optional[List[str]] = None) -> str:
    """
    Create a matplotlib confusion matrix visualization and save to file.
    Enhanced version supporting multiple use cases and MLflow integration.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        save_path: Full path to save file (takes precedence over filename)
        filename: Output filename (used with temp directory if save_path not provided)
        log_to_mlflow: Whether to log artifact to MLflow
        binary_conversion: How to convert to binary ('none', 'profitable_vs_rest', 'long_vs_rest')
        labels: Explicit labels for confusion matrix (defaults to unique values)
        label_names: Human-readable names for labels

    Returns:
        Path to saved plot file
    """
    # Convert to numpy arrays and handle NaN values
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Filter out NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    # Apply binary conversion if specified
    if binary_conversion == 'profitable_vs_rest':
        y_true_clean = (y_true_clean == 1).astype(int)
        y_pred_clean = (y_pred_clean == 1).astype(int)
        default_labels = [0, 1]
        default_label_names = ['Unprofitable', 'Profitable']
    elif binary_conversion == 'long_vs_rest':
        y_true_clean = (y_true_clean == 1).astype(int)
        y_pred_clean = (y_pred_clean == 1).astype(int)
        default_labels = [0, 1]
        default_label_names = ['Not Long', 'Long']
    else:
        default_labels = sorted(list(set(y_true_clean.tolist() + y_pred_clean.tolist())))
        default_label_names = [f'Class {i}' for i in default_labels]

    # Use provided labels or defaults
    if labels is None:
        labels = default_labels
    if label_names is None:
        if binary_conversion in ['profitable_vs_rest', 'long_vs_rest']:
            label_names = default_label_names
        else:
            label_names = [f'Class {i}' for i in labels]

    # Calculate confusion matrix
    cm = confusion_matrix(y_true_clean, y_pred_clean, labels=labels)

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Format tick labels
    if len(labels) == 2:
        x_labels = [f'Predicted\n{label_names[0]}', f'Predicted\n{label_names[1]}']
        y_labels = [f'Actual\n{label_names[0]}', f'Actual\n{label_names[1]}']
    else:
        x_labels = [f'Pred\n{name}' for name in label_names]
        y_labels = [f'Act\n{name}' for name in label_names]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=x_labels, yticklabels=y_labels,
                cbar_kws={'label': 'Number of Positions'})

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)

    # Add accuracy information
    accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
    plt.text(0.5, -0.15,
             f"Total Positions: {cm.sum()}\n"
             f"Accurate Predictions: {np.trace(cm)} ({accuracy*100:.1f}%)",
             transform=plt.gca().transAxes, ha='center', fontsize=10)

    # Determine save path
    if save_path is None:
        if filename is None:
            filename = "confusion_matrix.png"
        if not os.path.isabs(filename):
            save_path = os.path.join(tempfile.gettempdir(), filename)
        else:
            save_path = filename

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Log to MLflow if requested and available
    if log_to_mlflow and MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(save_path)
        except Exception as e:
            print(f"Warning: Could not log confusion matrix to MLflow: {e}")

    return save_path


def calculate_position_based_classification_metrics(signals: np.ndarray, 
                                                   returns: np.ndarray,
                                                   threshold: float = 0.0) -> Dict:
    """
    Calculate classification metrics based on period-level predictions.
    
    For long-short strategies:
    - Signal 1 = predicting positive returns
    - Signal -1 or 0 = predicting negative returns
    
    Args:
        signals: Array of trading signals (1 for long, -1 for short, 0 for cash)
        returns: Array of period returns 
        threshold: Threshold for defining profitable periods (default: 0.0)
        
    Returns:
        Dictionary containing classification metrics
    """
    # Filter out NaN values
    mask = ~(np.isnan(signals) | np.isnan(returns))
    clean_signals = signals[mask]
    clean_returns = returns[mask]
    
    if len(clean_signals) == 0:
        return {
            'num_periods': 0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'profitable_periods': 0,
            'unprofitable_periods': 0,
            'confusion_matrix': np.array([[0, 0], [0, 0]])
        }
    
    # Ground truth: 1 if return was positive, 0 if negative
    y_true = (clean_returns > threshold).astype(int)
    
    # Predictions: 1 if signal is long (1), 0 if signal is short (-1) or cash (0)
    y_pred = (clean_signals == 1).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Create confusion matrix plot
    import tempfile
    import os
    temp_dir = tempfile.gettempdir()
    cm_filename = os.path.join(temp_dir, f"confusion_matrix_{hash(str(y_true))}.png")
    cm_plot_path = create_confusion_matrix_plot(y_true, y_pred, filename=cm_filename)
    
    return {
        'num_periods': len(clean_signals),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'profitable_periods': sum(y_true),
        'unprofitable_periods': len(y_true) - sum(y_true),
        'long_predictions': sum(y_pred),
        'short_predictions': len(y_pred) - sum(y_pred),
        'confusion_matrix': cm,
        'confusion_matrix_plot': cm_plot_path
    }


def calculate_all_metrics(returns: np.ndarray,
                         signals: Optional[np.ndarray] = None,
                         benchmark_returns: Optional[np.ndarray] = None,
                         risk_free_rate: float = 0,
                         periods_per_year: int = 365) -> dict:
    """
    Calculate all evaluation metrics at once.
    
    Args:
        returns: Array of returns
        signals: Optional array of trading signals
        benchmark_returns: Optional array of benchmark returns for Information Ratio
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
    
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        'calmar_ratio': calculate_calmar_ratio(returns, periods_per_year),
        'max_drawdown': calculate_max_drawdown(returns),
        'annualized_volatility': calculate_annualized_volatility(returns, periods_per_year),
        'profit_factor': calculate_profit_factor(returns, signals),
        'avg_time_underwater': calculate_avg_time_underwater(returns),
        'total_return': (1 + returns).prod() - 1 if len(returns) > 0 else 0.0,
        'annualized_return': np.mean(returns) * periods_per_year if len(returns) > 0 else 0.0
    }
    
    # Add Information Ratio if benchmark returns are provided
    if benchmark_returns is not None:
        metrics['information_ratio'] = calculate_information_ratio(returns, benchmark_returns, periods_per_year)
    
    # Position-based classification removed - use triple barrier labels for ground truth instead
    
    return metrics


def export_comprehensive_csv(data, filename: str = "comprehensive_data.csv",
                            include_columns: Optional[List[str]] = None,
                            mlflow_log: bool = False) -> str:
    """
    Export comprehensive data to CSV with flexible column selection.

    Args:
        data: DataFrame or dict of data to export
        filename: Output filename
        include_columns: List of specific columns to include. If None, includes all.
        mlflow_log: Whether to log as MLflow artifact

    Returns:
        Path to saved CSV file
    """
    import pandas as pd
    import tempfile
    import os

    # Convert to DataFrame if not already
    if not isinstance(data, pd.DataFrame):
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        else:
            raise ValueError("Data must be DataFrame or dict")

    # Filter columns if specified
    if include_columns:
        available_columns = [col for col in include_columns if col in data.columns]
        if len(available_columns) != len(include_columns):
            missing = set(include_columns) - set(available_columns)
            print(f"Warning: Missing columns {missing}")
        data = data[available_columns]

    # Save to temporary file
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)
    data.to_csv(filepath)

    # Log to MLflow if requested
    if mlflow_log and MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(filepath)
            print(f"✅ Logged comprehensive CSV to MLflow: {filename}")
        except Exception as e:
            print(f"⚠️ Failed to log to MLflow: {e}")

    print(f"📊 Exported comprehensive data: {filepath}")
    print(f"   Rows: {len(data)}, Columns: {len(data.columns)}")

    return filepath


def create_model_summary(model, metrics: dict, feature_importance: Optional[dict] = None,
                        model_type: str = "unknown", save_path: Optional[str] = None,
                        mlflow_log: bool = False) -> str:
    """
    Create comprehensive model performance summary.

    Args:
        model: Trained model object
        metrics: Dictionary of performance metrics
        feature_importance: Optional feature importance dictionary
        model_type: Type of model (catboost, sklearn, linear, etc.)
        save_path: Optional path to save summary
        mlflow_log: Whether to log as MLflow artifact

    Returns:
        Path to saved summary file
    """
    import tempfile
    import os
    from datetime import datetime

    # Generate summary content
    summary_lines = [
        f"Model Performance Summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model Type: {model_type}",
        f"="*50,
        "",
        "PERFORMANCE METRICS:",
        "-"*20
    ]

    # Add metrics
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            summary_lines.append(f"{metric_name}: {value:.4f}")
        else:
            summary_lines.append(f"{metric_name}: {value}")

    summary_lines.append("")

    # Add model-specific information
    if hasattr(model, 'get_params'):
        try:
            params = model.get_params()
            summary_lines.extend([
                "MODEL PARAMETERS:",
                "-"*17
            ])
            for param, value in params.items():
                summary_lines.append(f"{param}: {value}")
            summary_lines.append("")
        except:
            pass

    # Add feature importance if provided
    if feature_importance:
        summary_lines.extend([
            "TOP 10 FEATURE IMPORTANCE:",
            "-"*26
        ])

        # Sort by importance and take top 10
        sorted_features = sorted(feature_importance.items(),
                               key=lambda x: abs(x[1]), reverse=True)[:10]

        for feature, importance in sorted_features:
            summary_lines.append(f"{feature}: {importance:.4f}")
        summary_lines.append("")

    # Add model info if available
    if hasattr(model, 'tree_count_'):
        summary_lines.append(f"Tree Count: {model.tree_count_}")
    elif hasattr(model, 'n_estimators'):
        summary_lines.append(f"N Estimators: {model.n_estimators}")

    # Create file path
    if save_path is None:
        temp_dir = tempfile.gettempdir()
        save_path = os.path.join(temp_dir, "model_summary.txt")

    # Write summary
    with open(save_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    # Log to MLflow if requested
    if mlflow_log and MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(save_path)
            print(f"✅ Logged model summary to MLflow")
        except Exception as e:
            print(f"⚠️ Failed to log to MLflow: {e}")

    print(f"📋 Created model summary: {save_path}")
    return save_path


def calculate_metrics_with_labels(
    predictions: np.ndarray,
    labels: np.ndarray,
    returns: Optional[np.ndarray] = None,
    plot_filename: str = "insample_label_confusion_matrix.png"
) -> dict:
    """
    Calculate classification metrics using triple barrier labels as ground truth.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions or strategy signals (-1, 0, 1)
    labels : np.ndarray
        Triple barrier labels (ground truth)
    returns : np.ndarray, optional
        Actual returns for additional metrics

    Returns
    -------
    dict
        Dictionary containing:
        - accuracy: Overall accuracy
        - precision: Precision for binary classification
        - recall: Recall for binary classification
        - f1: F1 score for binary classification
        - mcc: Matthews Correlation Coefficient
        - confusion_matrix_plot: Path to confusion matrix plot
    """
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                               accuracy_score, matthews_corrcoef)

    # Filter out NaN values
    mask = ~(np.isnan(predictions) | np.isnan(labels))
    clean_predictions = predictions[mask]
    clean_labels = labels[mask]

    if len(clean_predictions) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'mcc': 0.0,
            'confusion_matrix_plot': ''
        }

    # For binary classification: 1 = long, 0 = not long (short or neutral)
    binary_predictions = (clean_predictions == 1).astype(int)
    binary_labels = (clean_labels == 1).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(binary_labels, binary_predictions)
    precision = precision_score(binary_labels, binary_predictions, zero_division=0)
    recall = recall_score(binary_labels, binary_predictions, zero_division=0)
    f1 = f1_score(binary_labels, binary_predictions, zero_division=0)
    mcc = matthews_corrcoef(binary_labels, binary_predictions)

    # Create confusion matrix plot with custom filename
    temp_dir = tempfile.gettempdir()
    save_path = os.path.join(temp_dir, plot_filename)
    cm_plot_path = create_confusion_matrix_plot(binary_predictions, binary_labels, save_path=save_path)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'confusion_matrix_plot': cm_plot_path
    }

    # Add returns-based metrics if provided
    if returns is not None and len(returns) == len(clean_predictions):
        clean_returns = returns[mask]

        # Calculate returns for each predicted class
        long_returns = clean_returns[clean_predictions == 1]
        short_returns = -clean_returns[clean_predictions == -1]  # Invert for shorts

        metrics['avg_return_long'] = round(np.mean(long_returns), 4) if len(long_returns) > 0 else 0
        metrics['avg_return_short'] = round(np.mean(short_returns), 4) if len(short_returns) > 0 else 0

        # Hit rate: percentage of profitable trades
        if len(long_returns) > 0:
            metrics['hit_rate_long'] = round(np.sum(long_returns > 0) / len(long_returns), 4)
        else:
            metrics['hit_rate_long'] = 0

        if len(short_returns) > 0:
            metrics['hit_rate_short'] = round(np.sum(short_returns > 0) / len(short_returns), 4)
        else:
            metrics['hit_rate_short'] = 0

    return metrics