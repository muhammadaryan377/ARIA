"""ARIA Prescription Agent
--------------------------
Role: Recommended Actions

Turns the Insight Agent's findings (KPIs, trends, anomalies, hypotheses) plus
the Prediction Agent's forecast into 3-5 concrete, prioritized next actions a
manager should take. Uses the configured LLM (local Mistral or cloud Groq) and
falls back to a template derived from the actual numbers if the LLM fails.

Output: {"actions": [str, ...], "source": "llm" | "template"}
"""

import json
import logging
import re

from llm_provider import create_provider

logging.basicConfig(level=logging.INFO)


class PrescriptionAgent:
    def __init__(self, provider=None):
        self.llm = provider or create_provider()

    def prescribe(self, processed_data, kpis, trends, anomalies, hypotheses,
                  predictions=None, max_actions=5):
        """Return a dict with 'actions' (list of strings) and 'source'."""
        context = {
            "user_goal": processed_data.get("user_goal", "Unknown goal"),
            "row_count": processed_data.get("row_count", 0),
            "kpis": kpis[:6],
            "trends": trends[:3],
            "anomalies": anomalies[:5],
            "hypotheses": hypotheses[:4],
            "prediction": self._prediction_summary(predictions),
        }

        prompt = f"""
You are ARIA's Insight Agent. Recommend exactly {max_actions} concrete,
prioritized next actions for a non-technical manager. Tie each action to the
actual numbers in the context (KPIs, trends, anomalies, hypotheses, and the
forecast if one is present).

Context:
{json.dumps(context, indent=2, default=str)}

Rules:
- Output ONLY a numbered list, one action per line, no extra text.
- Each action must be specific enough to execute this week.
- If the forecast shows a strong projected change, include an action that
  prepares for it.
"""
        try:
            text = self.llm.chat(
                "story",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                num_predict=500,
            )
            actions = self._parse_actions(text, max_actions)
            if actions:
                return {"actions": actions, "source": "llm"}
        except Exception as exc:
            logging.warning("Prescription LLM failed: %s. Using template actions.", exc)

        return {
            "actions": self._template_actions(kpis, trends, anomalies, hypotheses,
                                              predictions, max_actions),
            "source": "template",
        }

    @staticmethod
    def _prediction_summary(predictions):
        # Only an accuracy-validated forecast may influence prescriptions.
        if not predictions or not predictions.get("valid") or not predictions.get("points"):
            return None
        return {
            "column": predictions["column"],
            "period": predictions["period"],
            "horizon": predictions["horizon"],
            "method": predictions["method"],
            "accuracy_pct": predictions["accuracy_pct"],
            "projected_pct_change": predictions["projected_pct_change"],
            "last_actual": predictions["last_actual"],
            "points": predictions["points"][:4],
        }

    @staticmethod
    def _parse_actions(text, max_actions):
        actions = []
        for line in str(text).splitlines():
            line = line.strip().replace("*", "").strip()
            if not line:
                continue
            m = re.match(r"^\s*(?:\d+[\.\)]\s*[\.\)\:]?|[-*]\s*|\u2022\s*)\s*(.*)$", line)
            if m and m.group(1).strip():
                actions.append(m.group(1).strip())
            if len(actions) >= max_actions:
                break
        if actions:
            return actions[:max_actions]

        # The model ignored the list format and returned prose. Split the text
        # into sentences so each actionable instruction survives as its own item.
        flat = re.sub(r"\s+", " ", str(text)).strip()
        if not flat:
            return []
        sentences = [s.strip().rstrip(".") for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()]
        if len(sentences) >= 2:
            return sentences[:max_actions]
        return [flat[:200]] if flat else []

    @staticmethod
    def _template_actions(kpis, trends, anomalies, hypotheses, predictions, max_actions):
        actions = []
        if hypotheses:
            actions.append(f"Validate the leading hypothesis: {hypotheses[0]['hypothesis']}")
        if trends:
            t = trends[0]
            actions.append(
                f"Review the {t['direction']} trend in {', '.join(t['columns'])} "
                f"(change {t['pct_change']}%)."
            )
        if anomalies:
            a = anomalies[0]
            actions.append(
                f"Investigate the anomaly in {a['column']} "
                f"(value {a['value']}, z-score {a['z_score']})."
            )
        if predictions and predictions.get("valid") and predictions.get("points"):
            period_word = predictions.get("period") or "periods"
            if period_word == "sequence":
                period_word = "periods"
            actions.append(
                f"Plan for the projected {predictions['projected_pct_change']}% change in "
                f"{predictions['column']} over the next {predictions['horizon']} {period_word} "
                f"(backtest accuracy {predictions['accuracy_pct']}% if shown)."
            )
        if kpis and not actions:
            top = max(kpis, key=lambda k: k["mean"])
            actions.append(f"Keep monitoring {top['column']} (mean {top['mean']}).")
        if not actions:
            actions.append("Continue monitoring the key metrics; no urgent action is indicated.")
        return actions[:max_actions]