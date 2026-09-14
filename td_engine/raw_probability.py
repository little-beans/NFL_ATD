from __future__ import annotations

import numpy as np
import pandas as pd


def _candidate_estimators(model):
    """
    Yield plausible pre-calibration estimators from common wrapper layouts.
    """
    seen = set()
    names = [
        "base",
        "base_model",
        "estimator",
        "classifier",
        "model",
        "clf",
        "pipeline",
    ]

    # Wrapper itself can sometimes be the estimator.
    objs = [("self", model)]
    for name in names:
        if hasattr(model, name):
            objs.append((name, getattr(model, name)))

    for name, obj in objs:
        if obj is None:
            continue
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        yield name, obj


def _candidate_feature_names(model, estimator):
    candidates = []

    for obj in [model, estimator]:
        if obj is None:
            continue

        for attr in [
            "features",
            "feature_names",
            "feature_columns",
            "model_features",
            "feature_names_in_",
        ]:
            if hasattr(obj, attr):
                value = getattr(obj, attr)
                if value is None:
                    continue
                try:
                    vals = list(value)
                except TypeError:
                    continue
                if vals:
                    candidates.append((attr, vals))

    # Deduplicate while preserving order.
    seen = set()
    result = []
    for source, vals in candidates:
        key = tuple(vals)
        if key not in seen:
            seen.add(key)
            result.append((source, vals))
    return result


def resolve_raw_probability(model, rows: pd.DataFrame):
    """
    Resolve a raw/pre-calibration TD probability from a saved model wrapper.

    Returns:
      probabilities: pandas Series
      diagnostics: dict

    Does not silently fill missing feature columns with zero unless the
    columns exist but contain NaN. If required columns are absent, the
    resolver reports them explicitly.
    """
    diagnostics = {
        "ok": False,
        "estimator_source": None,
        "feature_source": None,
        "features": [],
        "missing_features": [],
        "reason": None,
    }

    for est_name, estimator in _candidate_estimators(model):
        if not hasattr(estimator, "predict_proba"):
            continue

        feature_sets = _candidate_feature_names(model, estimator)

        # If no explicit features are stored, try sklearn feature_names_in_.
        if not feature_sets and hasattr(estimator, "feature_names_in_"):
            feature_sets = [("feature_names_in_", list(estimator.feature_names_in_))]

        # Last resort: if estimator exposes n_features_in_ and rows have exactly
        # that many numeric columns, do NOT guess ordering. Report instead.
        if not feature_sets:
            diagnostics["reason"] = (
                f"Estimator '{est_name}' supports predict_proba but no explicit "
                "feature-name list is available."
            )
            continue

        for feature_source, features in feature_sets:
            missing = [c for c in features if c not in rows.columns]

            if missing:
                diagnostics.update({
                    "estimator_source": est_name,
                    "feature_source": feature_source,
                    "features": features,
                    "missing_features": missing,
                    "reason": (
                        f"Weekly rows are missing {len(missing)} required model features."
                    ),
                })
                continue

            X = rows[features].copy()
            for c in features:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

            try:
                p = estimator.predict_proba(X)[:, 1]
            except Exception as exc:
                diagnostics.update({
                    "estimator_source": est_name,
                    "feature_source": feature_source,
                    "features": features,
                    "missing_features": [],
                    "reason": f"predict_proba failed: {exc}",
                })
                continue

            diagnostics.update({
                "ok": True,
                "estimator_source": est_name,
                "feature_source": feature_source,
                "features": features,
                "missing_features": [],
                "reason": None,
            })
            return pd.Series(p, index=rows.index, dtype=float), diagnostics

    return pd.Series(np.nan, index=rows.index, dtype=float), diagnostics
