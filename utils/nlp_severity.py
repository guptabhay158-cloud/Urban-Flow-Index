"""
UFI Project — NLP Severity Scorer
Rule-based keyword pipeline that converts unstructured incident text
into a normalised severity score in [0, 1].
"""

import re
import pandas as pd

# ── Keyword tiers ─────────────────────────────────────────────────────────────
TIER_WEIGHTS = {
    "high": 3,
    "medium": 2,
    "low": 1,
}

KEYWORDS = {
    "high": [
        "fatal", "fatality", "death", "killed", "blocked", "closed",
        "overturned", "collapsed", "explosion", "fire", "flood",
    ],
    "medium": [
        "accident", "collision", "crash", "construction", "repair",
        "breakdown", "stalled", "injured", "injuries", "signal failure",
        "diversion",
    ],
    "low": [
        "minor", "slow", "slight", "congestion", "delay", "queue",
        "fender", "pothole", "parking", "narrowing",
    ],
}

# Compile regex patterns (whole-word, case-insensitive)
PATTERNS = {
    tier: re.compile(
        r"\b(" + "|".join(re.escape(k) for k in words) + r")\b",
        re.IGNORECASE,
    )
    for tier, words in KEYWORDS.items()
}

MAX_POSSIBLE_SCORE = 5 * TIER_WEIGHTS["high"]   # ceiling for normalisation


def score_text(text: str) -> float:
    """
    Score a single incident description.

    Returns
    -------
    float
        Severity score normalised to [0, 1].
    """
    if not isinstance(text, str) or text.strip() == "":
        return 0.0

    raw = 0
    for tier, pattern in PATTERNS.items():
        matches = pattern.findall(text)
        raw += len(matches) * TIER_WEIGHTS[tier]

    return min(raw / MAX_POSSIBLE_SCORE, 1.0)


def score_dataframe(df: pd.DataFrame, text_col: str = "incident_text") -> pd.DataFrame:
    """
    Apply the scorer to every row of a DataFrame.
    Adds column ``nlp_severity`` in-place and returns the DataFrame.
    """
    df = df.copy()
    df["nlp_severity"] = df[text_col].apply(score_text)
    return df


# ── Demonstration ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "Fatal accident blocked all lanes near the intersection",
        "Minor fender bender cleared quickly",
        "Construction and signal failure causing major delays",
        "No incidents reported, traffic moving normally",
        "Truck overturned, road closed; two fatalities confirmed",
        "Slight congestion due to parking near market",
    ]

    print(f"{'Text':<60} {'Score':>6}")
    print("-" * 68)
    for s in samples:
        print(f"{s[:58]:<60} {score_text(s):>6.3f}")
