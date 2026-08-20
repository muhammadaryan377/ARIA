"""PredictionAgent - ARIA predictive analysis.

Turns a Goal-Agent `processed_data.json` payload into a structured forecast:

    {
      "column", "period", "horizon", "method",
      "accuracy_pct", "mape_pct", "reliability",
      "valid", "validation",   # gate: only valid=True may drive story/prescriptions/UI
      "model_notes", "last_actual",
      "projected_pct_change",
      "points": [{label, value, lower80, upper80, lower95, upper95}, ...]
    }

The numbers are produced by mathematical models (prediction/models.py) and
scored with honest out-of-sample walk-forward validation (prediction/evaluate.py),
so no generative model is involved in the actual forecast computation. Accuracy
is re-validated on every call, and if the backtested accuracy is below the
`valid_threshold` (or cannot be measured) the forecast is flagged invalid and is
not used by the business story, the prescriptions or the UI.
"""

import logging

import numpy as np
import pandas as pd

from .evaluate import accuracy_from_mape, reliability_label, select_model
from . import models as M

log = logging.getLogger("aria.prediction")


def _numeric_cols(df):
    """Numeric columns, including numeric-looking string columns (e.g. a Postgres
    NUMERIC value serialised as text by JSON)."""
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
            continue
        sample = df[c].dropna()
        if not len(sample):
            continue
        coerced = pd.to_numeric(sample, errors="coerce")
        if coerced.notna().sum() / len(sample) >= 0.8:
            cols.append(c)
    return cols


def _datetime_cols(df):
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() >= max(2, int(len(df) * 0.8)):
                cols.append(c)
        except Exception:
            continue
    return cols


def _infer_period(dates):
    """Choose the aggregation period from the median gap between dates."""
    series = pd.Series(pd.to_datetime(pd.unique(pd.to_datetime(dates, errors="coerce")))).sort_values().dropna()
    gaps = series.diff().dropna().dt.days
    if len(gaps) == 0:
        return "D"
    med = float(gaps.median())
    if med <= 2:
        return "D"
    if med <= 10:
        return "W"
    return "MS"


_PERIOD_NAMES = {"D": "daily", "W": "weekly", "MS": "monthly"}

# Column names that indicate a natural period/ordinal axis for forecasting even
# without a datetime column (e.g. "sales by year", "churn per month").
_PERIOD_KEYWORDS = {
    "period", "periods", "year", "years", "yr", "month", "months",
    "day", "days", "week", "weeks", "quarter", "quarters", "qtr",
    "seq", "sequence", "index", "idx", "order", "rank", "time",
    "date", "dates", "timestamp", "datetime",
}


def _forecast_labels(last_label, steps, period_code):
    if period_code is None:
        return [f"t+{k}" for k in range(1, steps + 1)]
    last_ts = pd.to_datetime(last_label)
    step = {
        "D": pd.Timedelta(days=1),
        "W": pd.Timedelta(weeks=1),
        "MS": pd.DateOffset(months=1),
    }[period_code]
    return [(last_ts + k * step).strftime("%Y-%m-%d") for k in range(1, steps + 1)]


class PredictionAgent:
    def __init__(self, horizon=6, min_points=5, valid_threshold=70.0):
        self.horizon = int(horizon)
        self.min_points = int(min_points)
        # Minimum out-of-sample accuracy (%) for a forecast to be presented as
        # valid. Below this the prediction is kept in the payload for transparency
        # but flagged invalid so the story, prescriptions and UI do not use it.
        self.valid_threshold = float(valid_threshold)

    def forecast(self, processed_data, numeric_col=None, datetime_col=None, horizon=None):
        horizon = horizon or self.horizon
        rows = (processed_data or {}).get("data") or []
        if not rows:
            return self._empty("empty dataset", "no_data")

        df = pd.DataFrame(rows)
        numeric_cols = _numeric_cols(df)
        if not numeric_cols:
            return self._empty("no numeric columns found", "no_data")
        # Coerce numeric-looking string columns to real numbers so downstream
        # aggregation and models compute correctly.
        for c in numeric_cols:
            if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
                try:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                except Exception:
                    pass

        if numeric_col is None:
            goal = str(processed_data.get("user_goal") or "").lower()
            # Identifier-like columns (ids/keys/codes) are never forecast targets:
            # they are synthetic keys, not measures.
            def _is_id_like(name):
                low = name.lower().strip()
                return low.endswith("_id") or low.endswith("_key") or low in ("id", "key", "code", "codeid")

            measure_cols = [c for c in numeric_cols if not _is_id_like(c)]
            if not measure_cols:
                # Only identifier-like numeric columns exist (e.g. an aggregate
                # that only selected keys). Forecasting a synthetic key is
                # meaningless, so degrade gracefully instead of guessing.
                return self._empty(
                    "no measure column to forecast (only id/key/code columns are numeric)",
                    "not_applicable",
                )
            numeric_col = next(
                (c for c in measure_cols if c.lower() in goal or goal in c.lower()),
                measure_cols[-1],  # measures usually come after id columns in SELECT order
            )
        if numeric_col not in df.columns:
            return self._empty(f"numeric column '{numeric_col}' not found", "no_data")

        if datetime_col is None:
            dcols = _datetime_cols(df)
            datetime_col = dcols[0] if dcols else None

        if datetime_col is not None:
            ts = pd.to_datetime(df[datetime_col], errors="coerce")
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_localize(None)
            mask = ts.notna()
            if mask.sum() >= 3:
                temp = df[mask].copy()
                temp["_ts"] = ts[mask]
                period_code = _infer_period(temp["_ts"])
                grouped = (
                    temp.groupby(pd.Grouper(key="_ts", freq=period_code))[numeric_col]
                    .sum()
                    .fillna(0.0)
                    .astype(float)
                )
                values = grouped.values
                labels = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in grouped.index]
                if len(values) >= 2:
                    season_period = self._season_period(period_code, len(values))
                    return self._build(
                        values, labels, numeric_col,
                        _PERIOD_NAMES[period_code], period_code, horizon, season_period,
                    )

        # No usable time column: a forecast is only meaningful when the rows carry
        # a natural period/ordinal axis (year, month, period, ...). Plain category
        # aggregates (e.g. "count by city") have no time dimension to forecast over
        # and are handled gracefully instead of being forced through a fake axis.
        seq_col = self._sequence_col(df)
        if seq_col is None:
            return self._empty(
                "no time dimension - a forecast is not applicable to this aggregate "
                f"question (forecast target '{numeric_col}').",
                "not_applicable",
            )

        seq = df[seq_col]
        values = pd.to_numeric(df[numeric_col], errors="coerce").astype(float).values
        mask = pd.notna(values)
        values = values[mask]
        seq = np.asarray(seq)[mask]
        if len(values) < 2:
            return self._empty(f"not enough numeric values in '{numeric_col}'", "insufficient_data")
        labels = [str(v) for v in seq.tolist()]
        return self._build(values, labels, numeric_col, "sequence", None, horizon, None)

    def _sequence_col(self, df):
        """Find a column that provides a natural period/ordinal axis."""
        for c in df.columns:
            low = str(c).lower().strip()
            if low in _PERIOD_KEYWORDS or low.startswith(("year", "month", "week", "qtr", "quarter", "period")):
                if df[c].notna().sum() >= 2 and df[c].nunique() >= 2:
                    return c
        return None

    def _season_period(self, period_code, n):
        """Season length for seasonal models, when enough history exists."""
        if period_code == "D" and n >= 21:
            return 7
        if period_code == "MS" and n >= 24:
            return 12
        if period_code == "W" and n >= 12:
            return 4
        return None

    def _build(self, values, labels, col, period_name, period_code, horizon, season_period=None):
        model_spec, oos_mape, oos_mae, meta = select_model(values, self.min_points, season_period)
        validated = model_spec is not None
        if not validated:
            # Too little history for walk-forward: fall back to a safe simple fit.
            model_spec = M.LinearTrendModel if len(values) >= 4 else M.NaiveModel
            meta = {"note": "insufficient history for walk-forward validation"}
            oos_mape = None
            oos_mae = None

        # An EnsembleModel arrives already fitted, but its .fit(values) is also a
        # no-op that records the series for fitted()/residuals(); classes are
        # fitted here on the full series.
        if isinstance(model_spec, type):
            model = model_spec().fit(values)
        else:
            model = model_spec.fit(values)
        steps = max(1, min(horizon, len(values)))
        points = np.asarray(model.predict(steps), dtype=float)

        resid = model.residuals(values)
        std = float(np.std(resid)) if len(resid) > 1 else 0.0
        # Prefer the honest out-of-sample error for the confidence band width;
        # fall back to in-sample std only when no walk-forward could be run.
        if oos_mae:
            # MAE ~= 0.8 * sigma for a Gaussian; rescale to an approximate sigma.
            sigma = oos_mae / 0.8
        else:
            sigma = std
        last_actual = float(values[-1])
        accuracy = accuracy_from_mape(oos_mape)
        projected_pct = (
            round(float(points[-1] / last_actual * 100.0 - 100.0), 2) if last_actual else 0.0
        )

        # Validation gate: only forecasts whose backtested accuracy clears the
        # threshold are "valid". Poor/unknown accuracy -> not usable downstream.
        valid = accuracy is not None and accuracy >= self.valid_threshold
        if not validated:
            reason = "insufficient history for walk-forward validation"
        elif accuracy is None:
            reason = "accuracy could not be measured (no walk-forward errors)"
        else:
            reason = (
                f"backtest accuracy {accuracy}% {'meets' if valid else 'below'} "
                f"the {self.valid_threshold}% threshold"
            )
        validation = {
            "method": "expanding-window walk-forward (one-step-ahead, out-of-sample)",
            "metric": "MAPE",
            "accuracy_pct": accuracy,
            "threshold_pct": self.valid_threshold,
            "passed": valid,
            "reason": reason,
        }

        fwd_labels = _forecast_labels(labels[-1], steps, period_code)
        forecast_points = [
            {
                "label": fwd_labels[i],
                "value": round(float(points[i]), 2),
                "lower80": round(float(points[i] - 1.28 * sigma), 2),
                "upper80": round(float(points[i] + 1.28 * sigma), 2),
                "lower95": round(float(points[i] - 1.96 * sigma), 2),
                "upper95": round(float(points[i] + 1.96 * sigma), 2),
            }
            for i in range(steps)
        ]

        return {
            "column": col,
            "period": period_name,
            "horizon": steps,
            "method": getattr(model, "name", "none"),
            "accuracy_pct": accuracy,
            "mape_pct": round(oos_mape, 2) if oos_mape is not None else None,
            "reliability": reliability_label(accuracy),
            "valid": valid,
            "reason": None if valid else reason,
            "validation": validation,
            "model_notes": meta,
            "last_actual": {"label": labels[-1], "value": round(last_actual, 2)},
            "projected_pct_change": projected_pct,
            "points": forecast_points,
        }

    def _empty(self, reason, reliability):
        return {
            "column": None, "period": None, "horizon": 0,
            "method": "none", "accuracy_pct": None, "mape_pct": None,
            "reliability": reliability, "reason": reason, "points": [],
            "valid": False,
            "validation": {
                "method": None, "metric": "MAPE",
                "accuracy_pct": None, "threshold_pct": self.valid_threshold,
                "passed": False, "reason": reason,
            },
            "last_actual": None, "projected_pct_change": None,
        }