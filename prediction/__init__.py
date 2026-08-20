"""ARIA predictive analysis package.

PredictionAgent turns processed data into a mathematically forecasted series
(with honest walk-forward accuracy) - see prediction/agent.py.
"""

from .agent import PredictionAgent

__all__ = ["PredictionAgent"]