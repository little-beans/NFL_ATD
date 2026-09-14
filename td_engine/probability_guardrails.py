from __future__ import annotations
import numpy as np
import pandas as pd

DEFAULT_MIN_PROB = 0.01
DEFAULT_MAX_PROB = 0.85


def clip_td_probability(
    values,
    min_prob: float = DEFAULT_MIN_PROB,
    max_prob: float = DEFAULT_MAX_PROB,
):
    """
    Guardrail for sparse calibration tails.

    This is NOT a replacement for proper calibration; it prevents
    production display/output from claiming literal 0% or 100% certainty.
    """
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    return s.clip(lower=min_prob, upper=max_prob)


def add_probability_diagnostics(
    df: pd.DataFrame,
    prob_col: str = "td_probability",
    min_prob: float = DEFAULT_MIN_PROB,
    max_prob: float = DEFAULT_MAX_PROB,
) -> pd.DataFrame:
    out = df.copy()
    if prob_col not in out.columns:
        return out

    raw = pd.to_numeric(out[prob_col], errors="coerce")
    out[f"{prob_col}_raw"] = raw
    out[prob_col] = raw.clip(min_prob, max_prob)
    out["probability_clipped"] = raw.ne(out[prob_col])
    return out
