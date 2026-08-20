"""
ARIA Insight Agent
------------------
Role: Signal Detection

Analyses processed_data.json (produced by the Goal Agent) for KPIs, trends,
and anomalies. Generates chart/dashboard specs, produces 3-4 ranked hypotheses
plus a human-readable business story (Mistral 7B locally, or Llama via Groq).

LLM backend is selected by the user at startup:
    local - Ollama (private, offline, slower)
    cloud - Groq API (fast, but data leaves the machine)

Output: insights.json

Usage:
    python insight_agent.py processed_data.json
"""

import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from llm_provider import LLMProvider, create_provider
from core.validation import validate_chart_choice
from prediction import PredictionAgent
from prescription_agent import PrescriptionAgent

logging.basicConfig(level=logging.INFO)


def sanitize_json(obj):
    """Recursively replace NaN/Inf floats with None.

    Statistic computations can produce NaN/Inf (e.g. 0/0 on a single-row
    dataset) which break strict JSON serializers like Starlette's
    JSONResponse (allow_nan=False). Applied to every agent payload before it
    is written or returned.
    """
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


CHART_TEMPLATES_PATH = Path(__file__).resolve().parent / "static" / "chart_templates.json"


class InsightAgent:
    def __init__(self, provider=None):
        self.llm = provider or create_provider()
        self.prediction = PredictionAgent()
        self.prescription = PrescriptionAgent(provider=self.llm)
        self._templates = None

    # ------------------------------------------------------------------
    # Chart templates (code/config kept in a JSON file, not in code)
    # ------------------------------------------------------------------

    def _load_chart_templates(self):
        """Load the reusable chart config templates from chart_templates.json."""
        if self._templates is None:
            if CHART_TEMPLATES_PATH.exists():
                try:
                    self._templates = json.loads(CHART_TEMPLATES_PATH.read_text(encoding="utf-8")).get("templates", {})
                except Exception:
                    self._templates = {}
            else:
                self._templates = {}
        return self._templates

    def _pick_template(self, templates, shape):
        """Pick the chart template whose rules best fit the data shape."""
        best_id, best_priority = None, -1
        for tid, tpl in templates.items():
            for rule in tpl.get("shapes", []):
                if rule.get("shape") == shape and rule.get("priority", 0) > best_priority:
                    best_priority = rule.get("priority", 0)
                    best_id = tid
        return best_id

    def _labels_datasets(self, xcol, ycol, records):
        return {
            "labels": [r.get(xcol) for r in records],
            "datasets": [{"label": ycol, "data": [r.get(ycol) for r in records]}],
        }

    # ------------------------------------------------------------------
    # Chart guardrail (data-driven rules decide; LLM never overrides)
    # ------------------------------------------------------------------

    def _guard_chart_type(self, default_tid, shape, suggestions, templates):
        """Return the template id to use for one data shape.

        The data-driven `default_tid` is authoritative. `suggestions` is kept
        for signature compatibility but is ignored by design: chart selection
        is fully deterministic so every provider renders the same charts.
        """
        if default_tid in templates:
            return default_tid
        return default_tid

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_processed_data(self, processed_data_path="processed_data.json"):
        with open(processed_data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Column classification
    # ------------------------------------------------------------------

    def _classify_columns(self, df):
        df = df.copy()
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        # Numeric-looking object columns (e.g. Decimal stringified by JSON, or
        # currencies with separators) must be treated as measures, not categories.
        for col in df.columns:
            if col in numeric:
                continue
            sample = df[col].dropna()
            if not len(sample):
                continue
            if sample.map(lambda v: isinstance(v, (datetime, pd.Timestamp))).all():
                continue
            coerced = pd.to_numeric(sample, errors="coerce")
            if int(coerced.notna().sum()) / len(sample) >= 0.8:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                numeric.append(col)

        category = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        datetime_cols = []
        for col in df.columns:
            if col in numeric:
                continue
            try:
                pd.to_datetime(df[col], errors="raise")
                if col in category:
                    category.remove(col)
                datetime_cols.append(col)
            except Exception:
                pass
        return numeric, category, datetime_cols

    # ------------------------------------------------------------------
    # KPI analysis
    # ------------------------------------------------------------------

    def compute_kpis(self, df, numeric_cols):
        """Compute descriptive KPIs for every numeric column."""
        kpis = []
        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.dropna().empty:
                continue
            count = int(series.count())
            kpis.append({
                "column": col,
                "sum": round(float(series.sum()), 2),
                "mean": round(float(series.mean()), 2),
                "median": round(float(series.median()), 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "std": (round(float(series.std()), 2) if count > 1 else None),
                "count": count,
                "null_count": int(series.isna().sum()),
            })
        return kpis

    # ------------------------------------------------------------------
    # Trend detection
    # ------------------------------------------------------------------

    def detect_trends(self, df, datetime_cols, numeric_cols):
        """Simple monotonic/up/down trend flag on time-series numeric columns."""
        trends = []
        for dt_col in datetime_cols[:1]:
            ts = pd.to_datetime(df[dt_col], errors="coerce")
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_localize(None)
            temp = df.copy()
            temp["_date"] = ts
            for num_col in numeric_cols:
                series = temp.groupby("_date")[num_col].sum().sort_index()
                if len(series) < 3:
                    continue
                values = series.astype(float).values
                diffs = np.diff(values)
                if np.all(diffs >= 0):
                    direction = "increasing"
                elif np.all(diffs <= 0):
                    direction = "decreasing"
                else:
                    direction = "mixed"
                pct_change = round(
                    float((values[-1] - values[0]) / values[0] * 100), 2
                ) if values[0] else 0.0
                trends.append({
                    "dimension": dt_col,
                    "direction": direction,
                    "pct_change": pct_change,
                    "columns": [num_col],
                })
        return trends

    # ------------------------------------------------------------------
    # Anomaly detection (z-score based)
    # ------------------------------------------------------------------

    def detect_anomalies(self, df, numeric_cols):
        """Flag numeric rows whose z-score exceeds the threshold as anomalies.
        Identifier-like columns (ids/keys/codes) are never flagged: they are
        synthetic keys, not measures."""
        anomalies = []
        for col in numeric_cols:
            low = col.lower().strip()
            if low.endswith("_id") or low.endswith("_key") or low in ("id", "key", "code", "codeid"):
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            valid = series.dropna()
            if len(valid) < 4:
                continue
            mean = valid.mean()
            std = valid.std()
            if std == 0 or pd.isna(std):
                continue
            z = (series - mean) / std
            for idx, value in series.items():
                if abs(z[idx]) > 2.0:
                    anomalies.append({
                        "column": col,
                        "row": int(idx),
                        "value": round(float(value), 2),
                        "z_score": round(float(z[idx]), 2),
                        "note": "beyond 2 standard deviations",
                    })
        return anomalies

    # ------------------------------------------------------------------
    # Chart / dashboard spec generation
    # ------------------------------------------------------------------

    def _histogram(self, series, bins=10):
        """Bin a numeric column into a histogram. Returns (labels, counts)."""
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0:
            return [], []
        lo, hi = float(s.min()), float(s.max())
        if not (hi > lo):
            return [], []
        edges = np.linspace(lo, hi, bins + 1)
        counts, _ = np.histogram(s.values, bins=edges)
        labels = [f"{edges[i]:,.1f}-{edges[i + 1]:,.1f}" for i in range(bins)]
        return labels, [int(c) for c in counts]

    @staticmethod
    def _deep_merge(base, extra):
        if not isinstance(base, dict) or not isinstance(extra, dict):
            return extra if extra is not None else base
        out = dict(base)
        for k, v in extra.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = InsightAgent._deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    def build_dashboard_spec(self, df, numeric_cols, category_cols, datetime_cols,
                             chart_suggestions=None):
        """Build chart specs by picking the chart template best suited to the data.

        Selection is data-intelligent: it considers category cardinality, the
        number of numeric series, the presence of a time axis, label lengths and
        distribution shape, then picks the best template from
        static/chart_templates.json. An LLM may propose chart types via
        `chart_suggestions`, but every proposal passes the deterministic guardrail
        (core.validation.validate_chart_choice) so the final type always fits the
        data shape and is identical on both providers.

        Each spec embeds a ready-to-render Chart.js `config`, so the frontend can
        add new chart types without touching client code.
        """
        templates = self._load_chart_templates()
        specs = []
        palette = ["#38bdf8", "#34d399", "#fbbf24", "#f87171", "#a78bfa", "#f472b6",
                   "#2dd4bf", "#facc15", "#60a5fa", "#fb7185"]

        def pick(shape, default):
            tid = self._pick_template(templates, shape) or default
            return self._guard_chart_type(tid, shape, chart_suggestions, templates)

        def pick_type(shape, preferred, default):
            """Pick a specific preferred type if it fits the shape, else the
            best-fit template. Lets a context force e.g. radar or bubble while
            the deterministic guardrail still decides validity."""
            if preferred:
                final_type, _ = validate_chart_choice(preferred, shape)
                if final_type == preferred and preferred in templates:
                    return preferred
            return pick(shape, default)

        def make(tid, title, x, y, labels, datasets, extra_options=None):
            cfg = templates.get(tid, {}).get("config", {})
            options = self._deep_merge(dict(cfg.get("options", {})), extra_options or {})
            options.setdefault("responsive", True)
            options.setdefault("maintainAspectRatio", False)
            chart_type = cfg.get("type", "bar")
            return {
                "chart_type": tid,
                "title": title,
                "x": x,
                "y": y,
                "labels": labels,
                "datasets": datasets,
                "template": cfg,
                "config": {
                    "type": chart_type,
                    "data": {"labels": labels, "datasets": datasets},
                    "options": options,
                },
            }

        def ds_style(label, data, kind, color):
            if kind in ("line", "area"):
                return {
                    "label": label, "data": data, "fill": kind == "area", "tension": 0.3,
                    "borderColor": color, "backgroundColor": color + "33",
                    "borderWidth": 2, "pointRadius": 3,
                }
            return {"label": label, "data": data, "backgroundColor": color,
                    "borderColor": color, "borderWidth": 1}

        def cat_summary(dim, col, topn):
            grouped = df.groupby(dim)[col].sum().sort_values(ascending=False).head(topn)
            return grouped.reset_index().to_dict(orient="records")

        # Identifier-like columns are keys, not measures: charts must plot real
        # business measures, never row IDs or foreign keys, unless those are the
        # only numeric columns available.
        def _is_id_like(name):
            low = name.lower().strip()
            return low.endswith("_id") or low.endswith("_key") or low in ("id", "key", "code", "codeid")

        measures_all = [c for c in numeric_cols if not _is_id_like(c)] or numeric_cols

        # ---- 1) Category -> numeric ----
        if category_cols and measures_all:
            dim = category_cols[0]
            try:
                card = int(df[dim].nunique())
            except Exception:
                card = 0
            primary = measures_all[0]

            recs = cat_summary(dim, primary, 10)
            if recs:
                labels = [r[dim] for r in recs]
                data = [float(r[primary]) for r in recs]
                longest = max((len(str(l)) for l in labels), default=0)
                if longest > 14 or card > 12:
                    tid = pick_type("category_numeric", "bar_horizontal", "bar")
                else:
                    tid = pick_type("category_numeric", "bar", "bar")
                specs.append(make(tid, f"{primary} by {dim} (top {len(labels)})", dim, primary,
                                  labels, [ds_style(primary, data, "bar", palette[0])]))

            # Share chart is only meaningful for a handful of categories.
            if 1 < card <= 6:
                top6 = cat_summary(dim, primary, 6)
                if top6:
                    tid = pick("category_low_card", "pie")
                    labels = [r[dim] for r in top6]
                    data = [float(r[primary]) for r in top6]
                    specs.append(make(
                        tid, f"{primary} share by {dim}", dim, primary, labels,
                        [{"label": primary, "data": data, "backgroundColor": palette,
                          "borderColor": "#1e293b", "borderWidth": 1}],
                    ))

            # Multiple numeric series over the same category -> stacked bars.
            if len(measures_all) >= 2:
                top8 = cat_summary(dim, primary, 8)
                if top8:
                    labels = [r[dim] for r in top8]
                    cols = measures_all[:4]
                    datasets = []
                    grouped = {c: df.groupby(dim)[c].sum() for c in cols}
                    for i, c in enumerate(cols):
                        ordered = [float(grouped[c].get(l, 0)) if l in grouped[c].index else 0.0
                                   for l in labels]
                        datasets.append(ds_style(c, ordered, "bar", palette[i % len(palette)]))
                    tid = pick("category_numeric_multi", "bar_stacked")
                    specs.append(make(tid, f"{' + '.join(cols)} by {dim} (top {len(labels)})",
                                      dim, "+".join(cols), labels, datasets))

            # Radar compares several numeric dimensions across few categories.
            if 2 <= card <= 8 and len(measures_all) >= 2:
                cats = df[dim].value_counts().head(8).index.tolist()
                cols = measures_all[:4]
                grouped = {c: df.groupby(dim)[c].sum() for c in cols}
                datasets = [
                    ds_style(c, [float(grouped[c].get(ct, 0)) for ct in cats], "line",
                             palette[i % len(palette)])
                    for i, c in enumerate(cols)
                ]
                tid = pick_type("category_numeric_multi", "radar", "radar")
                specs.append(make(tid, f"radar comparison across {dim}", dim, "+".join(cols),
                                  cats, datasets))

        # ---- 2) Datetime -> numeric (time series) ----
        if datetime_cols and measures_all:
            ts = pd.to_datetime(df[datetime_cols[0]], errors="coerce")
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_localize(None)
            temp = df.copy()
            temp["_date"] = ts
            temp = temp.dropna(subset=["_date"]).sort_values("_date")
            if not temp.empty:
                if len(measures_all) == 1:
                    col = measures_all[0]
                    line = temp.groupby("_date")[col].sum().reset_index()
                    labels = [str(d) for d in line["_date"]]
                    data = [float(v) for v in line[col]]
                    tid = pick("datetime_numeric", "line")
                    specs.append(make(tid, f"{col} over time", datetime_cols[0], col, labels,
                                      [ds_style(col, data, "line", palette[0])]))
                else:
                    cols = measures_all[:3]
                    grouped = {c: temp.groupby("_date")[c].sum() for c in cols}
                    labels = [str(d) for d in grouped[cols[0]].index]
                    datasets = []
                    for i, c in enumerate(cols):
                        ordered = [float(grouped[c].get(d, 0)) for d in grouped[cols[0]].index]
                        color = palette[i % len(palette)]
                        if i == 0:
                            datasets.append({**ds_style(c, ordered, "line", color),
                                             "yAxisID": "y", "type": "line"})
                        else:
                            datasets.append({**ds_style(c, ordered, "bar", color),
                                             "yAxisID": "y1"})
                    tid = pick("datetime_numeric_multi", "combo")
                    specs.append(make(tid, f"{' + '.join(cols)} over time", datetime_cols[0],
                                      "+".join(cols), labels, datasets))

        # ---- 3) Numeric columns only (no category, no time) ----
        if not category_cols and not datetime_cols and measures_all:
            measures = measures_all
            if len(measures) == 1:
                col = measures[0]
                labels, counts = self._histogram(df[col], bins=10)
                if labels:
                    tid = pick("distribution", "histogram")
                    specs.append(make(tid, f"distribution of {col}", col, "count", labels,
                                      [ds_style("count", counts, "bar", palette[0])]))
            else:
                x, y = measures[0], measures[1]
                sample = df[[x, y]].dropna().head(80)
                if not sample.empty:
                    tid = pick("numeric_numeric", "scatter")
                    pts = [{"x": float(a), "y": float(b)} for a, b in zip(sample[x], sample[y])]
                    specs.append(make(tid, f"{y} vs {x}", x, y, [],
                                      [{"label": y, "data": pts, "backgroundColor": palette[0]}]))
                if len(measures) >= 3:
                    z = measures[2]
                    s3 = df[[x, y, z]].dropna().head(60)
                    if not s3.empty and s3[z].abs().max() > 0:
                        tid2 = pick("numeric_numeric_numeric", "bubble")
                        scale = float(s3[z].abs().max()) or 1.0
                        bpts = [
                            {"x": float(a), "y": float(b),
                             "r": max(2.0, min(25.0, abs(float(c)) / scale * 25))}
                            for a, b, c in zip(s3[x], s3[y], s3[z])
                        ]
                        specs.append(make(
                            tid2, f"{z} as bubble size ({y} vs {x})", x, y, [],
                            [{"label": y, "data": bpts, "backgroundColor": palette[0] + "66",
                              "borderColor": palette[0], "borderWidth": 1}],
                        ))

        # ---- 4) Category only -> counts ----
        if category_cols and not numeric_cols and not datetime_cols:
            dim = category_cols[0]
            counts = df[dim].value_counts().head(10)
            if not counts.empty:
                labels = [str(i) for i in counts.index]
                data = [int(c) for c in counts.values]
                longest = max((len(str(l)) for l in labels), default=0)
                if longest > 14 or len(counts) > 12:
                    tid = pick_type("category_counts", "bar_horizontal", "bar")
                else:
                    tid = pick_type("category_counts", "bar", "bar")
                specs.append(make(tid, f"records by {dim} (top {len(labels)})", dim, "count",
                                  labels, [ds_style("count", data, "bar", palette[0])]))

        # ---- 5) KPI cards ----
        if numeric_cols:
            specs.append({
                "chart_type": "kpi_card",
                "title": "Key performance indicators",
                "x": None, "y": None, "labels": [], "datasets": [],
                "template": templates.get("kpi_card", {}).get("config", {}),
                "config": {"type": "kpi_card", "data": {"labels": [], "datasets": []}, "options": {}},
                "data": self.compute_kpis(df, numeric_cols),
            })

        return specs

    # ------------------------------------------------------------------
    # Hypothesis generation + ranking (statistical)
    # ------------------------------------------------------------------

    def generate_hypotheses(self, df, numeric_cols, category_cols, datetime_cols, kpis, anomalies):
        """360-degree analysis -> hypotheses in plain natural English.

        Every hypothesis is a sentence a business person can read directly.
        No technical jargon (no z-scores, std dev, grouped-sum terminology) in
        the output text or its rationale. Ranking is by `score`; `confidence`
        reflects how strongly the numbers support the claim.
        """
        hyps = []
        score = 0.0

        def add(text, basis, confidence, pts):
            nonlocal score
            hyps.append({
                "hypothesis": text,
                "basis": basis,
                "confidence": confidence,
                "score": float(pts),
            })
            score += float(pts)

        # Identifier-like columns are keys, not measures: hypotheses must talk
        # about real business measures, never about row IDs or foreign keys.
        def _is_id_like(name):
            low = name.lower().strip()
            return low.endswith("_id") or low.endswith("_key") or low in ("id", "key", "code", "codeid")

        measures = [c for c in numeric_cols if not _is_id_like(c)] or numeric_cols

        # 1) Across every numeric measure x every category dimension: who
        #    dominates, and is the value concentrated in a few groups?
        for col in measures:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue
            total = float(series.sum())
            if total == 0:
                continue
            for dim in category_cols[:2]:
                try:
                    grp = df.groupby(dim)[col].sum().dropna().sort_values(ascending=False)
                except Exception:
                    continue
                if grp.empty:
                    continue
                top = grp.index[0]
                top_share = float(grp.iloc[0]) / total * 100.0
                if top_share >= 20:
                    confidence = "high" if top_share >= 60 else ("medium" if top_share >= 35 else "low")
                    add(
                        f"'{top}' in {dim} contributes about {top_share:.0f}% of {col}, "
                        f"making it the dominant {dim} segment.",
                        f"Ranked every {dim} group by its share of total {col}.",
                        confidence, 10 + top_share / 10,
                    )
                top3_share = float(grp.head(3).sum()) / total * 100.0
                if top3_share >= 60 and len(grp) > 3:
                    add(
                        f"The top three {dim} groups together hold {top3_share:.0f}% of {col}, "
                        f"so {col} is heavily concentrated in a few {dim} groups.",
                        f"Measured how much of {col} sits in the three largest {dim} groups.",
                        "medium", 7,
                    )

        # 2) Compare the numeric measures with each other: which is the most
        #    volatile and which is the steadiest?
        if len(measures) >= 2:
            cv = []
            for col in measures:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s) and s.mean() != 0:
                    cv.append((col, float(s.std() / abs(s.mean()))))
            if cv:
                vol_col, vol = max(cv, key=lambda x: x[1])
                if vol > 0.5:
                    add(
                        f"{vol_col} swings the most relative to its average, so it is the "
                        f"measure that most deserves close monitoring.",
                        "Compared how much each measure varies relative to its own average.",
                        "medium", 6,
                    )
                stable_col, stab = min(cv, key=lambda x: x[1])
                if stab < 0.15 and len(cv) > 1:
                    add(
                        f"{stable_col} stays very close to its average throughout, making it "
                        f"the most predictable measure in the data.",
                        "Compared how much each measure varies relative to its own average.",
                        "low", 4,
                    )

        # 3) Outliers / anomalies described in plain language.
        by_col = {}
        for a in anomalies:
            if _is_id_like(a.get("column", "")):
                continue
            by_col.setdefault(a["column"], []).append(a)
        for col, items in list(by_col.items())[:2]:
            n = len(items)
            add(
                f"{col} contains {n} {'observation' if n == 1 else 'observations'} that "
                f"{'stands' if n == 1 else 'stand'} well apart from the typical range and "
                f"{'is' if n == 1 else 'are'} worth investigating.",
                "Flagged the rows whose value is far outside the normal spread of the column.",
                "medium", 6 + n,
            )

        # 4) Time dimension present.
        if datetime_cols:
            add(
                f"The figures change over time (along {datetime_cols[0]}), which may explain "
                f"recent moves in the key numbers.",
                "Detected a time axis and checked how the measures move along it.",
                "low", 4,
            )

        # 5) Anchor on the descriptive statistics.
        if kpis:
            top = max(kpis, key=lambda k: k.get("mean") or 0)
            add(
                f"On average, {top['column']} sits around {top['mean']:,.2f}, typically "
                f"ranging from {top.get('min')} to {top.get('max')}.",
                "Summarised the descriptive statistics of each measure.",
                "high", 5,
            )

        hyps.sort(key=lambda h: h["score"], reverse=True)
        return hyps[:5], score

    # ------------------------------------------------------------------
    # Business story via Mistral 7B
    # ------------------------------------------------------------------

    def generate_business_story(self, processed_data, kpis, trends, anomalies, hypotheses,
                                predictions=None):
        """Return a short, plain-English business story for the findings.

        A deterministic narrative engine always produces the story from the
        actual numbers (identical on cloud and local, and never blocks). When
        the LLM is available it is asked to polish/expand that narrative; any
        failure, timeout or unusable output falls back to the deterministic
        version so the story is never missing or stuck.
        """
        base = self._narrative_story(processed_data, kpis, trends, anomalies, hypotheses, predictions)

        context = {
            "user_goal": processed_data.get("user_goal", "Unknown goal"),
            "row_count": processed_data.get("row_count", 0),
            "kpis": kpis[:6],
            "trends": trends[:3],
            "anomalies": anomalies[:5],
            "hypotheses": hypotheses,
        }
        if predictions and predictions.get("valid") and predictions.get("points"):
            context["prediction"] = {
                "column": predictions["column"],
                "horizon": predictions["horizon"],
                "method": predictions["method"],
                "accuracy_pct": predictions["accuracy_pct"],
                "projected_pct_change": predictions["projected_pct_change"],
                "points": predictions["points"][:4],
            }

        prompt = f"""
You are the Insight Agent of a BI system. Below is a draft narrative written from
the data, followed by the raw findings. Rewrite it as a polished, human-readable
business story for a non-technical manager.

Draft narrative:
{base}

Raw findings:
{json.dumps(context, indent=2, default=str)}

Rules:
- Keep every claim grounded in the numbers given; do not invent facts.
- Start with the headline finding, then supporting points, then any outlook.
- Plain language, under 250 words, no markdown, no JSON, no lists of jargon.
"""
        try:
            polished = self.llm.chat("story", messages=[{"role": "user", "content": prompt}],
                                     temperature=0.4, num_predict=400, timeout=45)
            polished = (polished or "").strip()
            if len(polished) >= 40 and len(polished) <= 1200:
                return self._plain_english_story(polished)
            logging.warning("Story LLM output unusable; using deterministic narrative.")
        except Exception as exc:
            logging.warning(f"Storytelling LLM failed: {exc}. Using deterministic narrative.")
        return base

    def _plain_english_story(self, text):
        """Strip residual technical jargon an LLM polish may have reintroduced,
        so the story stays readable for a non-technical manager."""
        import re as _re
        subs = [
            (r"\bstandard deviation\b", "typical variation"),
            (r"\bstandard deviations\b", "typical variations"),
            (r"\bmedian\b", "middle value"),
            (r"\bcoefficient of variation\b", "relative variation"),
            (r"\bdispersion\b", "spread"),
            (r"\bz-score\b", "deviation score"),
            (r"\bstats\b", "statistics"),
        ]
        out = text
        for pat, repl in subs:
            out = _re.sub(pat, repl, out, flags=_re.IGNORECASE)
        return out

    def _narrative_story(self, processed_data, kpis, trends, anomalies, hypotheses,
                         predictions=None):
        """Deterministic, plain-English narrative assembled from the findings.
        Always available and identical for every provider."""
        goal = (processed_data.get("user_goal") or "the business question").strip()
        rows = processed_data.get("row_count", 0)
        parts = []
        if goal.lower() != "unknown goal":
            parts.append(f"This analysis answers the question: {goal} — based on "
                         f"{rows} data {('row' if rows == 1 else 'rows')}.")
        else:
            parts.append(f"This analysis covers {rows} data {('row' if rows == 1 else 'rows')}.")

        if hypotheses:
            parts.append(f"The headline finding is that {hypotheses[0]['hypothesis']}")
        if kpis:
            top = max(kpis, key=lambda k: k.get("mean") or 0)
            parts.append(
                f"On average, {top['column']} is about {top['mean']:,.2f}, "
                f"ranging from {top.get('min')} to {top.get('max')}."
            )
        if trends:
            t = trends[0]
            parts.append(
                f"{', '.join(t['columns'])} shows a {t['direction']} movement over time, "
                f"a change of {t['pct_change']}%."
            )
        for h in hypotheses[1:3]:
            parts.append(f"We also observe that {h['hypothesis']}")
        if anomalies:
            a = anomalies[0]
            parts.append(
                f"One value deserves attention: a {a['column']} of {a['value']} sits well "
                f"outside the usual range and is flagged for a closer look."
            )
        if predictions and predictions.get("valid") and predictions.get("points"):
            period_word = predictions.get("period") or "periods"
            if period_word in ("sequence",):
                period_word = "periods"
            parts.append(
                f"Looking ahead, {predictions['column']} is projected to change by about "
                f"{predictions['projected_pct_change']}% over the next "
                f"{predictions['horizon']} {period_word}."
            )
        story = " ".join(parts)
        return story or "No notable signals detected in this dataset."

    # ------------------------------------------------------------------
    # Main analysis pipeline
    # ------------------------------------------------------------------

    def analyze(self, processed_data_path="processed_data.json", output_path="insights.json"):
        warnings = []
        try:
            processed_data = self.load_processed_data(processed_data_path)
            rows = processed_data.get("data", [])
            df = pd.DataFrame(rows)
        except Exception as exc:
            logging.warning("Insight Agent could not load processed data: %s", exc)
            processed_data = {}
            df = pd.DataFrame()
            warnings.append(f"load_processed_data: {exc}")

        print(f"Insight Agent analysing {processed_data_path} "
              f"({processed_data.get('row_count', len(df))} rows)...")

        def safe(step_name, fn, default):
            try:
                return fn()
            except Exception as exc:
                logging.warning("Insight step '%s' failed: %s", step_name, exc)
                warnings.append(f"{step_name}: {exc}")
                return default

        numeric_cols, category_cols, datetime_cols = safe(
            "classify_columns", lambda: self._classify_columns(df), ([], [], [])
        )
        # Force numeric columns to real numbers (they may arrive as strings or
        # Decimals) so every downstream step computes correctly for any source.
        for col in numeric_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    pass
        kpis = safe("compute_kpis", lambda: self.compute_kpis(df, numeric_cols), [])
        trends = safe("detect_trends", lambda: self.detect_trends(df, datetime_cols, numeric_cols), [])
        anomalies = safe("detect_anomalies", lambda: self.detect_anomalies(df, numeric_cols), [])
        chart_suggestions = None
        dashboard_spec = safe(
            "build_dashboard_spec",
            lambda: self.build_dashboard_spec(
                df, numeric_cols, category_cols, datetime_cols, chart_suggestions
            ),
            [],
        )
        hypotheses, ranking_score = safe(
            "generate_hypotheses",
            lambda: self.generate_hypotheses(
                df, numeric_cols, category_cols, datetime_cols, kpis, anomalies
            ),
            ([], 0.0),
        )

        predictions = None
        if numeric_cols:
            try:
                predictions = self.prediction.forecast(
                    processed_data,
                    datetime_col=datetime_cols[0] if datetime_cols else None,
                )
            except Exception as exc:
                logging.warning("Prediction Agent failed: %s", exc)
                warnings.append(f"prediction: {exc}")

        prescriptions = None
        try:
            prescriptions = self.prescription.prescribe(
                processed_data, kpis, trends, anomalies, hypotheses, predictions
            )
        except Exception as exc:
            logging.warning("Prescription Agent failed: %s", exc)
            warnings.append(f"prescription: {exc}")

        story = safe(
            "generate_business_story",
            lambda: self.generate_business_story(
                processed_data, kpis, trends, anomalies, hypotheses, predictions
            ),
            None,
        )
        if not story:
            try:
                story = self._narrative_story(processed_data, kpis, trends, anomalies,
                                              hypotheses, predictions)
            except Exception as exc:
                logging.warning("Template story fallback failed: %s", exc)
                story = (
                    "ARIA analysed the data, but the narrative could not be "
                    "generated. Review the KPIs, charts, and recommendations below."
                )

        insights = {
            "generated_at": datetime.now().isoformat(),
            "source_processed_data": processed_data_path,
            "user_goal": processed_data.get("user_goal", "Unknown goal"),
            "summary": {
                "rows": int(len(df)),
                "columns": list(df.columns),
                "numeric_columns": numeric_cols,
                "categorical_columns": category_cols,
                "datetime_columns": datetime_cols,
            },
            "kpis": kpis,
            "trends": trends,
            "anomalies": anomalies,
            "hypotheses": hypotheses,
            "predictions": predictions,
            "prescriptions": prescriptions,
            "hypothesis_ranking_score": round(ranking_score, 2),
            "dashboard": dashboard_spec,
            "business_story": story,
            "warning": "; ".join(warnings) if warnings else None,
        }

        out_path = Path(output_path)
        out_path.write_text(
            json.dumps(sanitize_json(insights), indent=2, default=str), encoding="utf-8"
        )
        print(f"Insight Agent done! Saved to {output_path}")
        return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python insight_agent.py processed_data.json")
        sys.exit(1)

    agent = InsightAgent()
    agent.analyze(sys.argv[1])
