# Soros System Signal Framework

This directory contains the signal framework for the Soros System, which provides a modular and extensible way to create, evaluate, and combine trading signals.

## Overview

The signal framework consists of the following components:

1. **Signal Base Interface**: Defines the common interface that all signals must implement
2. **Signal Registry**: Central registry for managing available signals
3. **Signal Implementations**: Various signal implementations based on different indicators and strategies
4. **Signal Evaluation**: Tools for evaluating signal effectiveness using forward returns analysis
5. **Signal Combination**: Tools for combining multiple signals using optimal weights

## Usage

### Creating a Signal

To create a new signal, inherit from the `SignalBase` class and implement the required methods:

```python
from soros_system.signals import SignalBase, register_signal

@register_signal
class MySignal(SignalBase):
    def __init__(self, params=None):
        super().__init__(params)
        # Initialize your signal parameters
        
    def validate(self, data, asset_id):
        # Validate that the data has what we need
        return True
        
    def calculate(self, data, asset_id):
        # Implement signal calculation logic
        # Return a Series with signal values (+1 or -1)
```

### Using Signals

To use signals in your code:

```python
from soros_system.signals import get_signal, get_all_signals
from soros_system.portfolio import SignalCombiner

# Get all available signals
available_signals = get_all_signals()

# Create signal instances
my_signal = get_signal('MySignal', params={'param1': 'value1'})
another_signal = get_signal('AnotherSignal')

# Create signal combiner
combiner = SignalCombiner()

# Combine signals
combined_signal, final_decision, weights = combiner.combine_signals(
    [my_signal, another_signal], data, asset_id
)
```

## Available Signals

### Trend Signals

- **ShortTermTrendSignal**: Signal based on short-term trend classification
- **MediumTermTrendSignal**: Signal based on medium-term trend classification
- **LongTermTrendSignal**: Signal based on long-term trend classification
- **OverallTrendSignal**: Signal based on overall trend classification

### RSI Signals

- **RSISignal**: Signal based on RSI crossing specific thresholds
- **RSIWithRoCSignal**: Signal based on RSI and Rate of Change combined

### Volatility Signals

- **MarkovVolatilitySignal**: Signal based on Markov volatility model regime detection
- **VolatilityTrendSignal**: Signal based on the trend of volatility changes

## Signal Evaluation

The signal framework includes tools for evaluating signal effectiveness:

```python
from soros_system.analysis.forward_returns import SignalEvaluator

# Create signal evaluator
evaluator = SignalEvaluator(
    periods=[1, 3, 5, 7, 14, 21, 28],  # Forward periods to evaluate
    min_samples=30,                    # Minimum samples required
    lookback_days=365                  # Use last year of data
)

# Evaluate signals
evaluations = evaluator.evaluate_multiple_signals(signals, data, asset_id)

# Calculate weights
weights = evaluator.calculate_normalized_weights(evaluations)
```

## Examples

See the `examples/signal_framework_example.py` script for a complete example of how to use the signal framework.

## Extending the Framework

To add new signal types, simply create a new class that inherits from `SignalBase` and implement the required methods. Then register it using the `@register_signal` decorator.

The framework is designed to be easily extended with new signal types, evaluation methods, and combination strategies. 