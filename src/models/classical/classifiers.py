"""Classical scikit-learn classifiers for network fingerprinting."""

from __future__ import annotations

from typing import Any, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#: Name -> scikit-learn estimator class.
MODELS = {
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "linear": LogisticRegression,
    "knn": KNeighborsClassifier,
    "svm": SVC,
}

#: Default hyperparameters (overridden by user-supplied ``params``).
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "decision_tree": {},
    "random_forest": {"n_estimators": 100},
    "linear": {"max_iter": 1000},
    "knn": {"n_neighbors": 5},
    "svm": {},
}


def available_models() -> tuple[str, ...]:
    """Return the names of the supported classifiers."""
    return tuple(MODELS)


def build_classifier(name: str, params: Optional[dict[str, Any]] = None):
    """Instantiate a scikit-learn estimator by name.

    ``params`` override the model's default hyperparameters.
    """
    if name not in MODELS:
        raise ValueError(
            f"Unknown model {name!r}; expected one of {available_models()}"
        )
    merged = {**DEFAULT_PARAMS.get(name, {}), **(params or {})}
    return MODELS[name](**merged)
