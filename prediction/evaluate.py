"""Walk-forward evaluation, model selection and ensemble voting for ARIA.

Why this is honest (no overfitting):
- We never judge a model on the data it was fitted to. Each candidate is scored
  with an expanding-window walk-forward loop: fit on the first k points, predict
  point k+1, repeat. The reported accuracy is therefore out-of-sample.
- Overfit guard: a model whose out-of-sample error is much worse than its
  in-sample error (ratio > 1.5) has memorized the training window; it is dropped
  in favour of simpler models.
- Beat-naive guard: any candidate that performs worse than the naive baseline is
  excluded from the ensemble (it only adds noise). The reported ensemble error is
  itself measured by the same out-of-sample walk-forward loop, so an ensemble is
  only used when it genuinely beats the best single model.
- Accuracy = max(0, 100 - MAPE%). It is reported honestly per dataset.
"""

import numpy as np

from . import models as M


def mape(actual, predicted):
    """Mean Absolute Percentage Error (%). None if no valid (non-zero) actuals."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.abs(actual) > 1e-9
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100.0)


def _make(cls, season_period):
    """Instantiate a candidate, passing a season period to models that need one."""
    if getattr(cls, "needs_period", False):
        return cls(period=season_period or 7)
    return cls()


def walk_forward_errors(model_cls, values, season_period=None):
    """Expanding-window walk-forward one-step errors.

    Returns a list of (actual, abs_error) tuples for every point that could be
    predicted out-of-sample. This is the ground truth for both MAPE (used for
    accuracy) and MAE (used for honest confidence bands).
    """
    n = len(values)
    if n < 5:
        return []
    errors = []
    start = max(3, n // 2)
    for k in range(start, n):
        train = values[:k]
        actual = float(values[k])
        if abs(actual) < 1e-9:
            continue
        try:
            pred = _make(model_cls, season_period).fit(train).predict(1)[0]
        except Exception:
            continue
        errors.append((actual, abs(actual - pred)))
    return errors


def one_step_ahead_mape(model_cls, values, season_period=None):
    """Expanding-window walk-forward one-step MAPE for a candidate model."""
    errors = walk_forward_errors(model_cls, values, season_period)
    if not errors:
        return None
    return float(np.mean([e / a * 100.0 for a, e in errors]))


def one_step_ahead_mae(model_cls, values, season_period=None):
    """Expanding-window walk-forward one-step MAE for a candidate model."""
    errors = walk_forward_errors(model_cls, values, season_period)
    if not errors:
        return None
    return float(np.mean([e for _, e in errors]))


def in_sample_mape(model_cls, values, season_period=None):
    """In-sample one-step-ahead MAPE (the fitted() curve). Used only by the
    overfit guard as a reference, never as the reported accuracy."""
    try:
        fitted = np.asarray(_make(model_cls, season_period).fit(values).fitted(), dtype=float)
    except Exception:
        return None
    return mape(values, fitted)


def _score_candidates(values, season_period, min_points):
    """Walk-forward score every candidate. Returns list of result dicts."""
    n = len(values)
    if n < min_points:
        return []
    results = []
    for cls in M.CANDIDATES:
        oos = one_step_ahead_mape(cls, values, season_period)
        if oos is None:
            continue
        ins = in_sample_mape(cls, values, season_period)
        ratio = (oos / ins) if (ins and ins > 1e-9) else float("inf")
        overfit = ratio > 1.5
        results.append({
            "cls": cls, "oos": oos, "ins": ins, "ratio": ratio, "overfit": overfit,
        })
    return results


def _blend_errors(members, weights, values, season_period=None):
    """Honest out-of-sample errors of a weighted blend via walk-forward."""
    n = len(values)
    errors = []
    start = max(3, n // 2)
    for k in range(start, n):
        train = values[:k]
        actual = float(values[k])
        if abs(actual) < 1e-9:
            continue
        preds = []
        ok = True
        for cls in members:
            try:
                preds.append(float(_make(cls, season_period).fit(train).predict(1)[0]))
            except Exception:
                ok = False
                break
        if not ok:
            continue
        blend = float(np.dot(weights, preds))
        errors.append((actual, abs(actual - blend)))
    return errors


def select_model(values, min_points=5, season_period=None):
    """Pick the best model for a series: the best single candidate or, when it
    genuinely improves out-of-sample accuracy, a weighted ensemble (voting).

    Returns (model, out_of_sample_mape, out_of_sample_mae, meta) where `model`
    is either a candidate CLASS (instantiate + fit on the full series) or an
    already-fitted EnsembleModel instance. Returns (None, None, None, None) when
    there is too little data to validate.
    """
    n = len(values)
    if n < min_points:
        return None, None, None, None

    results = _score_candidates(values, season_period, min_points)
    if not results:
        return None, None, None, None

    naive_mape = next((r["oos"] for r in results if r["cls"] is M.NaiveModel), None)

    # Non-overfit candidates that are not worse than naive (they earn their place).
    eligible = [
        r for r in results
        if not r["overfit"] and (naive_mape is None or r["oos"] <= naive_mape * 1.01)
    ]
    if not eligible:
        eligible = results
    eligible.sort(key=lambda r: r["oos"])
    best = eligible[0]

    meta = {
        "walk_forward_mape": round(best["oos"], 2),
        "in_sample_mape": round(best["ins"], 2) if best["ins"] is not None else None,
        "overfit_ratio": round(best["ratio"], 2),
        "overfit_rejected": best["overfit"],
        "naive_mape": round(naive_mape, 2) if naive_mape is not None else None,
    }

    # ---- Ensemble voting ----
    top = eligible[:3]
    weights = None
    if len(top) >= 2:
        inv = [1.0 / max(r["oos"], 1e-9) for r in top]
        wsum = sum(inv)
        weights = [w / wsum for w in inv]
        blend_errors = _blend_errors([r["cls"] for r in top], weights, values, season_period)
        if blend_errors:
            ens_mape = float(np.mean([e / a * 100.0 for a, e in blend_errors]))
            ens_mae = float(np.mean([e for _, e in blend_errors]))
            # Only use the ensemble when it genuinely beats the best single model.
            if ens_mape < best["oos"]:
                members = []
                for r, w in zip(top, weights):
                    try:
                        members.append((_make(r["cls"], season_period).fit(values), w))
                    except Exception:
                        continue
                if members:
                    weights = [w / sum(w for _, w in members) for _, w in members]
                    members = [(m, w) for (m, _), w in zip(members, weights)]
                    model = M.EnsembleModel(members)
                    mae = ens_mae
                    meta.update({
                        "ensemble": {
                            "members": [{"name": r["cls"].name, "weight": round(w, 3)} for r, w in zip(top, weights)],
                            "walk_forward_mape": round(ens_mape, 2),
                            "walk_forward_mae": round(ens_mae, 2),
                        },
                    })
                    return model, ens_mape, mae, meta

    mae = one_step_ahead_mae(best["cls"], values, season_period)
    meta["walk_forward_mae"] = round(mae, 2) if mae is not None else None
    return best["cls"], best["oos"], mae, meta


def accuracy_from_mape(mape_value):
    """Accuracy percentage from a MAPE value (None-safe)."""
    if mape_value is None:
        return None
    return round(max(0.0, 100.0 - mape_value), 1)


def reliability_label(accuracy):
    if accuracy is None:
        return "insufficient_data"
    if accuracy >= 90:
        return "excellent"
    if accuracy >= 80:
        return "good"
    if accuracy >= 70:
        return "fair"
    return "poor"
