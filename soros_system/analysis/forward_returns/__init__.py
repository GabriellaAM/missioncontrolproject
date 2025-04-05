"""
Forward returns analysis package for Soros System.

This package contains tools for calculating and analyzing forward return distributions
to evaluate signal effectiveness.
"""

from .calculator import ForwardReturnsCalculator
from .statistical_tests import StatisticalTester
from .signal_evaluator import SignalEvaluator

__all__ = [
    'ForwardReturnsCalculator',
    'StatisticalTester',
    'SignalEvaluator'
] 