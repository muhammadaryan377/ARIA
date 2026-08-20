"""Forecasting math models for ARIA predictive analysis.

Every model is a tiny, dependency-light estimator built on numpy only:

- Naive / Mean: cheap baselines.
- Moving average: short-window smoothing.
- Linear trend: least-squares straight-line fit (numpy polyfit).
- Exponential smoothing: level-only, alpha chosen by grid search.
- Holt linear (damped trend): level + damped trend, alpha/beta by grid search.

Models expose a uniform interface:
    fit(values) -> self
    predict(steps) -> np.ndarray (point forecasts)
    fitted() -> np.ndarray (in-sample one-step-ahead fits, same length as values)
    residuals(values) -> np.ndarray (values - fitted)
"""

import numpy as np


class BaseModel:
    name = "base"

    def fit(self, values):
        raise NotImplementedError

    def predict(self, steps):
        raise NotImplementedError

    def fitted(self):
        raise NotImplementedError

    def residuals(self, values):
        values = np.asarray(values, dtype=float)
        return values - np.asarray(self.fitted(), dtype=float)


class NaiveModel(BaseModel):
    """Forecast = the last observed value. The hard baseline to beat."""

    name = "naive"

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.values = values
        self.last = float(values[-1]) if len(values) else 0.0
        return self

    def predict(self, steps):
        return np.full(steps, self.last)

    def fitted(self):
        if len(self.values) == 0:
            return np.array([])
        fitted = np.empty(len(self.values))
        fitted[0] = self.values[0]
        fitted[1:] = self.values[:-1]
        return fitted


class MeanModel(BaseModel):
    """Forecast = the historical mean. Baseline for flat series."""

    name = "mean"

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.values = values
        self.mean = float(values.mean()) if len(values) else 0.0
        return self

    def predict(self, steps):
        return np.full(steps, self.mean)

    def fitted(self):
        return np.full(len(self.values), self.mean)


class MovingAverageModel(BaseModel):
    """Short-window rolling mean; flat forecast at the last smoothed value."""

    name = "moving_average"

    def __init__(self, window=3):
        self.window = int(window)

    def fit(self, values):
        self.values = np.asarray(values, dtype=float)
        return self

    def _smooth(self):
        n = len(self.values)
        out = np.empty(n)
        for i in range(n):
            lo = max(0, i - self.window + 1)
            out[i] = self.values[lo:i + 1].mean()
        return out

    def predict(self, steps):
        if len(self.values) == 0:
            return np.full(steps, 0.0)
        return np.full(steps, self._smooth()[-1])

    def fitted(self):
        return self._smooth()


class LinearTrendModel(BaseModel):
    """Least-squares linear trend via numpy polyfit."""

    name = "linear_trend"

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.values = values
        n = len(values)
        t = np.arange(n, dtype=float)
        if n >= 2:
            self.coef = np.polyfit(t, values, 1)
        else:
            self.coef = np.array([0.0, values[0] if n else 0.0])
        return self

    def predict(self, steps):
        n = len(self.values)
        t = np.arange(n, n + steps, dtype=float)
        return np.polyval(self.coef, t)

    def fitted(self):
        n = len(self.values)
        return np.polyval(self.coef, np.arange(n, dtype=float))


class ExponentialSmoothingModel(BaseModel):
    """Level-only exponential smoothing; alpha grid-searched to minimize SSE.

    fitted() uses proper one-step-ahead forecasts (state from the previous step),
    so the in-sample errors reflect real predictive skill, not smoothing.
    """

    name = "exponential_smoothing"

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.values = values
        n = len(values)
        best_alpha, best_sse = 0.5, np.inf
        for alpha in np.arange(0.05, 1.0, 0.05):
            fitted, level = self._run(alpha)
            sse = float(np.sum((values - fitted) ** 2))
            if sse < best_sse:
                best_alpha, best_sse = alpha, sse
        self.alpha = best_alpha
        self._fitted, self.level = self._run(self.alpha)
        return self

    def _run(self, alpha):
        n = len(self.values)
        fitted = np.empty(n)
        if n == 0:
            return fitted, 0.0
        level = float(self.values[0])
        fitted[0] = level
        for i in range(1, n):
            fitted[i] = level
            level = alpha * float(self.values[i]) + (1 - alpha) * level
        return fitted, level

    def predict(self, steps):
        return np.full(steps, self.level)

    def fitted(self):
        return self._fitted


class HoltLinearModel(BaseModel):
    """Holt's linear method with a damped trend (phi < 1) to avoid explosive
    long-horizon extrapolation. alpha/beta grid-searched to minimize SSE of
    one-step-ahead fits."""

    name = "holt_linear_damped"

    def __init__(self, damp=True, phi=0.9):
        self.damp = damp
        self.phi = float(phi)

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.values = values
        best = (0.3, 0.1, np.inf)
        for alpha in np.arange(0.1, 1.0, 0.1):
            for beta in np.arange(0.05, 0.55, 0.05):
                fitted, _, _ = self._run(alpha, beta)
                sse = float(np.sum((values - fitted) ** 2))
                if sse < best[2]:
                    best = (alpha, beta, sse)
        self.alpha, self.beta = best[0], best[1]
        self._fitted, self._level, self._trend = self._run(self.alpha, self.beta)
        return self

    def _run(self, alpha, beta):
        n = len(self.values)
        fitted = np.empty(n)
        if n == 0:
            return fitted, 0.0, 0.0
        level = float(self.values[0])
        trend = float(self.values[1] - self.values[0]) if n > 1 else 0.0
        fitted[0] = level
        for i in range(1, n):
            forecast = level + self.phi * trend
            fitted[i] = forecast
            new_level = alpha * float(self.values[i]) + (1 - alpha) * forecast
            trend = beta * (new_level - level) + (1 - beta) * self.phi * trend
            level = new_level
        return fitted, level, trend

    def predict(self, steps):
        out = np.empty(steps)
        level, trend = self._level, self._trend
        for i in range(steps):
            trend = self.phi * trend
            level = level + trend
            out[i] = level
        return out

    def fitted(self):
        return self._fitted


class MedianModel(BaseModel):
    """Forecast = the historical median. Robust baseline for flat / skewed series."""

    name = "median"

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.values = values
        self.median = float(np.median(values)) if len(values) else 0.0
        return self

    def predict(self, steps):
        return np.full(steps, self.median)

    def fitted(self):
        return np.full(len(self.values), self.median)


class DriftModel(BaseModel):
    """Forecast by projecting the average drift from the first to the last value."""

    name = "drift"

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.values = values
        n = len(values)
        self.start = float(values[0]) if n else 0.0
        self.drift = (float(values[-1]) - self.start) / (n - 1) if n > 1 else 0.0
        return self

    def predict(self, steps):
        n = len(self.values)
        return self.start + self.drift * (n - 1 + np.arange(1, steps + 1))

    def fitted(self):
        n = len(self.values)
        return self.start + self.drift * np.arange(n)


class SeasonalNaiveModel(BaseModel):
    """Forecast each step as the value observed one full season earlier.

    The natural baseline for strongly seasonal series (e.g. day-of-week or
    month-of-year patterns)."""

    name = "seasonal_naive"
    needs_period = True

    def __init__(self, period=7):
        self.period = max(2, int(period))

    def fit(self, values):
        self.values = np.asarray(values, dtype=float)
        return self

    def predict(self, steps):
        v = self.values
        p = self.period
        n = len(v)
        if n == 0:
            return np.zeros(steps)
        out = np.empty(steps)
        for i in range(steps):
            idx = n - p + ((i) % p)
            out[i] = v[max(0, min(n - 1, idx))]
        return out

    def fitted(self):
        v = self.values
        p = self.period
        n = len(v)
        out = np.empty(n)
        for i in range(n):
            out[i] = v[i - p] if i >= p else v[i]
        return out


class HoltWintersModel(BaseModel):
    """Additive Holt-Winters (level + trend + season) with grid-searched
    alpha/beta/gamma. Season period is required and is set by the caller based on
    the detected frequency (e.g. 7 for daily, 12 for monthly)."""

    name = "holt_winters_additive"
    needs_period = True

    def __init__(self, period=7):
        self.period = max(2, int(period))

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        self.values = values
        n = len(values)
        best = None
        for alpha in np.arange(0.1, 0.9, 0.1):
            for beta in np.arange(0.05, 0.45, 0.1):
                for gamma in np.arange(0.05, 0.45, 0.1):
                    try:
                        fitted = self._run(alpha, beta, gamma)[0]
                        sse = float(np.sum((values - fitted) ** 2))
                    except Exception:
                        continue
                    if best is None or sse < best[0]:
                        best = (sse, alpha, beta, gamma)
        if best is None:
            self.alpha, self.beta, self.gamma = 0.3, 0.1, 0.1
        else:
            self.alpha, self.beta, self.gamma = best[1], best[2], best[3]
        self._fitted, self._level, self._trend, self._season = self._run(
            self.alpha, self.beta, self.gamma
        )
        return self

    def _run(self, alpha, beta, gamma):
        values = self.values
        n = len(values)
        p = self.period
        if n == 0:
            return np.array([]), 0.0, 0.0, np.zeros(p)
        season = np.zeros(p)
        m = min(p, n)
        first_mean = float(values[:m].mean())
        season[:m] = values[:m] - first_mean
        level = first_mean
        trend = 0.0
        fitted = np.empty(n)
        for i in range(n):
            s_idx = i % p
            fitted[i] = level + trend + season[s_idx]
            new_level = alpha * (float(values[i]) - season[s_idx]) + (1 - alpha) * (level + trend)
            trend = beta * (new_level - level) + (1 - beta) * trend
            season[s_idx] = gamma * (float(values[i]) - new_level) + (1 - gamma) * season[s_idx]
            level = new_level
        return fitted, level, trend, season

    def predict(self, steps):
        out = np.empty(steps)
        level, trend = self._level, self._trend
        p = self.period
        n = len(self.values)
        for i in range(steps):
            out[i] = level + trend + self._season[(n + i) % p]
            level = level + trend
        return out

    def fitted(self):
        return self._fitted


class EnsembleModel(BaseModel):
    """Weighted blend of already-fitted member models (voting ensemble).

    Created by prediction/evaluate.py; `fit(values)` is a no-op because members
    are fitted externally on the full series."""

    name = "ensemble"

    def __init__(self, fitted_members=None):
        # fitted_members: list of (fitted_model_instance, weight)
        self.fitted_members = fitted_members or []
        self.values = np.array([])
        names = "+".join(m.name for m, _ in self.fitted_members[:3])
        if len(self.fitted_members) > 3:
            names += "+"
        self.name = f"ensemble({names})" if names else "ensemble"

    def fit(self, values):
        self.values = np.asarray(values, dtype=float)
        return self

    def predict(self, steps):
        preds = np.zeros(steps)
        for m, w in self.fitted_members:
            preds = preds + w * np.asarray(m.predict(steps), dtype=float)
        return preds

    def fitted(self):
        if len(self.values) == 0:
            return np.array([])
        preds = np.zeros(len(self.values))
        for m, w in self.fitted_members:
            preds = preds + w * np.asarray(m.fitted(), dtype=float)
        return preds

    def residuals(self, values):
        values = np.asarray(values, dtype=float)
        return values - np.asarray(self.fitted(), dtype=float)


# Candidate models for model selection, roughly simplest -> most complex.
CANDIDATES = [
    NaiveModel,
    MeanModel,
    MedianModel,
    DriftModel,
    MovingAverageModel,
    LinearTrendModel,
    ExponentialSmoothingModel,
    HoltLinearModel,
    SeasonalNaiveModel,
    HoltWintersModel,
]