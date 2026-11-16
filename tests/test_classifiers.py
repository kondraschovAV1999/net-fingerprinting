import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from models.classical.classifiers import available_models, build_classifier


def test_available_models():
    assert set(available_models()) == {
        "decision_tree",
        "random_forest",
        "linear",
        "knn",
        "svm",
    }


def test_build_returns_correct_estimator_type():
    assert isinstance(build_classifier("decision_tree"), DecisionTreeClassifier)
    assert isinstance(build_classifier("random_forest"), RandomForestClassifier)
    assert isinstance(build_classifier("linear"), LogisticRegression)
    assert isinstance(build_classifier("knn"), KNeighborsClassifier)
    assert isinstance(build_classifier("svm"), SVC)


def test_build_applies_params():
    clf = build_classifier("knn", {"n_neighbors": 3})
    assert clf.n_neighbors == 3


def test_default_params_applied():
    assert build_classifier("random_forest").n_estimators == 100
    assert build_classifier("linear").max_iter == 1000
    assert build_classifier("knn").n_neighbors == 5


def test_params_override_defaults():
    assert build_classifier("random_forest", {"n_estimators": 10}).n_estimators == 10


def test_build_unknown_model_raises():
    with pytest.raises(ValueError):
        build_classifier("nope")


def test_every_model_fits_and_predicts():
    rng = np.random.RandomState(0)
    X = rng.rand(24, 5)
    y = np.array(["a", "b", "c"] * 8)
    for name in available_models():
        clf = build_classifier(name)
        clf.fit(X, y)
        pred = clf.predict(X[:6])
        assert len(pred) == 6
        assert set(pred).issubset({"a", "b", "c"})
