"""Database-agnostic validation of user goals (rejects gibberish / noise)."""

import json
import re

from fastapi import HTTPException

_GOAL_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "been", "being", "to", "of", "in", "on", "for", "with",
    "at", "by", "from", "it", "this", "that", "these", "those", "as", "how",
    "why", "when", "where", "who", "what", "which", "show", "tell", "give",
    "me", "please", "hi", "hello", "do", "does", "can", "could", "would",
    "should", "the", "your", "our", "there", "their", "you", "you're",
    "yours", "yourself", "am", "we", "we're", "us", "them", "they", "they're",
    "i", "i'm", "im", "he", "she", "it's", "its", "his", "her",
}

_GIBSMASH = {
    "asdf", "asdfg", "asdfgh", "asdfghj", "fdsa", "sadf", "dsf", "fds",
    "qwerty", "qwer", "wert", "poiu", "poiuy", "rewq", "ytrewq",
    "sdfg", "dfgh", "fghj", "ghjk", "hjkl", "jkl", "jkl;", "klj", "kjhg",
    "zxcv", "zxcvb", "xcvbn", "cvbnm", "mnbvc", "vbnm", "nbvc", "bvcx",
    "aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff", "gggg", "hhhh", "iiii",
    "jjjj", "kkkk", "llll", "mmmm", "nnnn", "oooo", "pppp", "qqqq", "rrrr",
    "ssss", "tttt", "uuuu", "vvvv", "wwww", "xxxx", "yyyy", "zzzz",
    "asdfjkl", "qwe", "rty", "uio", "p", "sdf", "fgh", "jkl", "lkj",
    "test", "testing", "test123", "abc", "abcd", "abcde", "xyz", "xy",
    "lorem", "ipsum", "loremipsum", "fuck", "shit", "poop", "butt",
}

_GOAL_VOWELS = set("aeiouy")


def validate_goal_text(goal: str) -> str:
    """Lightweight, database-agnostic validation.

    Rejects empty input, pure noise, keyboard smashes and obvious gibberish,
    while accepting any real business question regardless of database domain.
    """
    cleaned = (goal or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Please write a business question, e.g. 'Show total sales by category'.")

    if not re.search(r"[A-Za-z]", cleaned):
        raise HTTPException(status_code=400, detail="That doesn't look like a real question. Please describe what you want in words.")

    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", cleaned.lower())
    if len(tokens) < 2:
        raise HTTPException(status_code=400, detail="Please write a fuller business question, e.g. 'Which product has the highest revenue?'")

    # Single-letter tokens ("a b c d e f g", "x y") add no meaning: a real
    # question needs at least a couple of proper words.
    real_words = [t for t in tokens if len(t) >= 2]
    if len(real_words) < 2:
        raise HTTPException(status_code=400, detail="That doesn't look like a real question. Please type a meaningful business goal.")

    for token in tokens:
        if re.fullmatch(r"(.)\1{3,}", token):
            raise HTTPException(status_code=400, detail="That doesn't look like a real question. Please type a meaningful business goal.")
        if len(token) >= 4 and len(set(token)) <= 2:
            raise HTTPException(status_code=400, detail="That doesn't look like a real question. Please type a meaningful business goal.")
        if len(token) >= 4 and not any(ch in _GOAL_VOWELS for ch in token):
            raise HTTPException(status_code=400, detail="That doesn't look like a real question. Please type a meaningful business goal.")
        if token in _GIBSMASH:
            raise HTTPException(status_code=400, detail="That doesn't look like a real question. Please type a meaningful business goal.")

    content_words = [t for t in real_words if t not in _GOAL_STOPWORDS]
    if not content_words:
        raise HTTPException(status_code=400, detail="That doesn't look like a real question. Please type a meaningful business goal.")

    return cleaned


# ---------------------------------------------------------------------------
# LLM output guardrail layer (provider-agnostic).
#
# Every agent that consumes raw model output (local Ollama or cloud Groq) must
# run it through these helpers BEFORE trusting it. Local models are weaker and
# more prone to echo, refusals, code-fence soup and gibberish, so a single
# shared layer keeps the behaviour identical on both providers.
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:sql|json|python|text)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

_REFUSAL_RE = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bi\s+(?:cannot|cant|can't|won't|am\s+unable|don't\s+know|am\s+sorry|apologize)",
        r"\bas\s+an\s+ai",
        r"i'm\s+(?:sorry|not\s+able)",
        r"i\s+don't\s+have\s+(?:access|enough)",
        r"cannot\s+(?:provide|help|assist)",
    )
]

_SQL_TABLE_RE = re.compile(
    r"\b(?:FROM|JOIN|UPDATE|INTO|DELETE\s+FROM|TABLE)\s+[\"`\[]?([A-Za-z_][\w$]*)", re.IGNORECASE
)


def strip_code_fences(text):
    """Extract the content of any ```...``` block, otherwise return trimmed text."""
    text = (text or "").strip()
    if not text:
        return ""
    blocks = _CODE_FENCE_RE.findall(text)
    if blocks:
        return max((b.strip() for b in blocks), key=len)
    return text


def looks_like_refusal(text):
    low = (text or "").lower()
    return any(r.search(low) for r in _REFUSAL_RE)


def is_gibberish(text, min_letters=6):
    t = strip_code_fences(text or "")
    if not t:
        return True
    if re.fullmatch(r"([A-Za-z0-9])\1{4,}", t):
        return True
    letters = [ch for ch in t if ch.isalpha()]
    if len(letters) < min_letters:
        return True
    if len(set(letters)) / len(letters) < 0.35:
        return True
    return False


def clean_llm_text(raw):
    """Validate and normalise free-text model output.

    Returns (text, None) on success or (None, reason) when the output is
    empty / a refusal / gibberish and must not be trusted.
    """
    text = strip_code_fences(raw or "")
    if not text:
        return None, "empty output"
    if looks_like_refusal(text):
        return None, "refusal detected"
    if is_gibberish(text):
        return None, "gibberish output"
    return text, None


def parse_llm_json(raw):
    """Validate JSON model output (accepts code-fenced or embedded JSON).

    Returns (parsed, None) or (None, reason).
    """
    text = strip_code_fences(raw or "")
    if not text:
        return None, "empty output"
    try:
        return json.loads(text), None
    except Exception:
        pass
    block = re.search(r"\{.*\}", text, re.DOTALL)
    if block:
        try:
            return json.loads(block.group(0)), None
        except Exception:
            pass
    return None, "invalid JSON"


def sql_table_names(sql):
    """Return the set of table identifiers referenced by a SQL statement."""
    names = set()
    for m in _SQL_TABLE_RE.finditer(sql or ""):
        name = m.group(1)
        if name.lower() not in ("dual",):
            names.add(name)
    return names


def unknown_sql_tables(sql, schema_tables):
    """Return referenced tables that do not exist in the schema.

    This catches hallucinations like `order_items` when the real table is
    `order_details` — deterministically, before a database round-trip.
    """
    known = {t.lower() for t in (schema_tables or [])}
    return sorted(n for n in sql_table_names(sql) if n.lower() not in known)


def empty_sql_tables(sql, table_counts):
    """Return tables referenced by `sql` that exist but contain zero rows."""
    counts = {t.lower(): c for t, c in (table_counts or {}).items()}
    return sorted(n for n in sql_table_names(sql) if counts.get(n.lower(), 1) == 0)


# ---------------------------------------------------------------------------
# Column-level SQL validation.
#
# Catches hallucinated COLUMN references (e.g. `orders.quantity` when the real
# column lives in `order_details`). Deterministic and provider-agnostic, so a
# weak local model and the cloud model are held to the same standard. FROM/JOIN
# aliases are resolved back to their real table names before checking.
# ---------------------------------------------------------------------------

_SQL_ALIAS_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+[\"`\[]?([A-Za-z_][\w$]*)[\]\"`]?"
    r"(?:\s+(?:AS\s+)?[\"`\[]?([A-Za-z_][\w$]*)[\]\"`]?)?",
    re.IGNORECASE,
)

_SQL_QUALIFIED_RE = re.compile(
    r"\b([A-Za-z_][\w$]*)\s*\.\s*[\"`\[]?([A-Za-z_][\w$]*)[\]\"`]?",
    re.IGNORECASE,
)


def sql_qualified_columns(sql):
    """Return the set of `table.column` references in a SQL statement.

    FROM/JOIN aliases are resolved back to their real table name first, so both
    `orders.quantity` and `o.quantity` (with `FROM orders o`) yield the same
    reference. Qualifiers that cannot be resolved to any alias are returned
    verbatim; bare columns (no qualifier) are not returned.
    """
    aliases = {}
    for m in _SQL_ALIAS_RE.finditer(sql or ""):
        table = m.group(1)
        alias = m.group(2)
        aliases[table.lower()] = table
        if alias:
            aliases[alias.lower()] = table
    refs = set()
    for m in _SQL_QUALIFIED_RE.finditer(sql or ""):
        qual = m.group(1)
        col = m.group(2)
        refs.add(f"{aliases.get(qual.lower(), qual)}.{col}")
    return refs


def unknown_sql_columns(sql, columns_by_table):
    """Return `table.column` references that do not exist in the schema.

    `columns_by_table` maps a table name to an iterable of its real columns.
    Only references whose qualifier resolves to a KNOWN table are validated;
    unresolvable qualifiers and bare columns are skipped (the former is the
    job of `unknown_sql_tables`). This catches hallucinations like
    `orders.quantity` when quantity only exists on `order_details`.
    """
    columns_by_table = {
        t.lower(): {c.lower() for c in cols}
        for t, cols in (columns_by_table or {}).items()
    }
    bad = []
    for ref in sorted(sql_qualified_columns(sql)):
        table, _, col = ref.partition(".")
        if table.lower() not in columns_by_table:
            continue
        if col.lower() not in columns_by_table[table.lower()]:
            bad.append(ref)
    return bad


# ---------------------------------------------------------------------------
# Chart-type guardrail.
#
# The LLM (local or cloud) may PROPOSE a chart type after seeing the data, but
# a deterministic rule decides the final type based on the data shape. Both
# providers pass through the exact same check, so charts are identical whether
# the backend is Ollama or Groq and a weak local model can never force a chart
# that does not fit the data.
# ---------------------------------------------------------------------------

CHART_WHITELIST = {
    "bar", "bar_horizontal", "bar_stacked", "line", "area", "combo",
    "scatter", "bubble", "pie", "doughnut", "polar_area", "radar",
    "histogram", "kpi_card",
}

CHART_SHAPE_RULES = {
    "category_numeric": {
        "allowed": {"bar", "bar_horizontal", "line", "pie", "doughnut"},
        "default": "bar",
    },
    "category_numeric_multi": {
        "allowed": {"bar_stacked", "bar", "radar", "line", "combo"},
        "default": "bar_stacked",
    },
    "category_low_card": {
        "allowed": {"pie", "doughnut", "polar_area", "bar"},
        "default": "pie",
    },
    "category_counts": {
        "allowed": {"bar", "bar_horizontal", "pie", "doughnut"},
        "default": "bar",
    },
    "datetime_numeric": {
        "allowed": {"line", "area", "bar", "combo"},
        "default": "line",
    },
    "datetime_numeric_multi": {
        "allowed": {"combo", "line", "area", "bar_stacked"},
        "default": "combo",
    },
    "numeric_numeric": {
        "allowed": {"scatter", "bubble", "line"},
        "default": "scatter",
    },
    "numeric_numeric_numeric": {
        "allowed": {"bubble", "scatter"},
        "default": "bubble",
    },
    "distribution": {
        "allowed": {"histogram", "bar"},
        "default": "histogram",
    },
    "share": {
        "allowed": {"pie", "doughnut", "polar_area", "bar"},
        "default": "pie",
    },
}

_CHART_ALIASES = {
    "bar": "bar", "bars": "bar", "column": "bar", "columns": "bar",
    "horizontal bar": "bar_horizontal", "hbar": "bar_horizontal", "barh": "bar_horizontal",
    "stacked bar": "bar_stacked", "stacked": "bar_stacked",
    "line": "line", "lines": "line", "trend": "line", "line chart": "line",
    "area": "area", "area chart": "area",
    "combo": "combo", "combination": "combo", "dual axis": "combo", "dual-axis": "combo",
    "scatter": "scatter", "scatterplot": "scatter", "scatter plot": "scatter",
    "bubble": "bubble", "bubble chart": "bubble",
    "pie": "pie", "pie chart": "pie",
    "donut": "doughnut", "doughnut": "doughnut", "donut chart": "doughnut",
    "polar": "polar_area", "polar area": "polar_area", "radar": "radar",
    "histogram": "histogram", "hist": "histogram",
    "kpi": "kpi_card", "card": "kpi_card", "metric": "kpi_card", "cards": "kpi_card",
}


def _normalize_chart(suggested):
    """Map a free-form LLM suggestion (e.g. 'Pie chart!') to a whitelisted id."""
    if not suggested:
        return None
    low = str(suggested).strip().lower()
    if low in CHART_WHITELIST:
        return low
    if low in _CHART_ALIASES:
        return _CHART_ALIASES[low]
    for word in re.findall(r"[a-z]+", low):
        if word in _CHART_ALIASES:
            return _CHART_ALIASES[word]
    return None


def validate_chart_choice(suggested, shape):
    """Deterministic guardrail for an LLM-proposed chart type.

    Returns (chart_type, reason). A suggestion is used only if it is a real
    chart type AND fits the data shape; otherwise the shape's default wins.
    """
    rule = CHART_SHAPE_RULES.get(shape)
    if not rule:
        return "bar", "unknown shape -> default 'bar'"
    default = rule["default"]
    norm = _normalize_chart(suggested)
    if norm in rule["allowed"]:
        return norm, f"accepted '{norm}' for {shape}"
    if norm:
        return default, f"rejected '{norm}' for {shape} (does not fit) -> default '{default}'"
    return default, f"no usable suggestion for {shape} -> default '{default}'"
