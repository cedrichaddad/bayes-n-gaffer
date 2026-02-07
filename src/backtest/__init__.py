"""
Backtest module for Alpha-FPL.

Implements walk-forward backtesting with rolling origin evaluation.
"""

from .runner import BacktestRunner

__all__ = [
    "BacktestRunner",
]
