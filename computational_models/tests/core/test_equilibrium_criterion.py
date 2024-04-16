import pytest

from computational_models.core.equilibrium_criterion import EquilibriumCriterion, WithoutCriterion


def test_without_criterion():
    criterion = WithoutCriterion()
    assert not criterion.in_equilibrium([])
    assert not criterion.in_equilibrium([1])
    assert not criterion.in_equilibrium([1, 2])
    assert not criterion.in_equilibrium([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert not criterion.in_equilibrium([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    balanced = criterion.in_equilibrium(
        [1, 2.99, 3.00, 3.01, 3.00, 3.01, 3.01, 3.02, 3.03, 3.02, 3.01, 3.02]
    )
    assert not balanced


def test_equilibrium_criterion_without_series_name_raises_exception():
    with pytest.raises(TypeError):
        EquilibriumCriterion()


def test_equilibrium_criterion_with_default_values():
    criterion = EquilibriumCriterion(series_name="test_series")
    assert criterion.series_name == "test_series"
    assert criterion.window_size == 20
    assert criterion.tolerance == 0.001


def test_equilibrium_criterion_with_custom_values():
    criterion = EquilibriumCriterion("test_series", 10, 0.01)
    assert criterion.series_name == "test_series"
    assert criterion.window_size == 10
    assert criterion.tolerance == 0.01


def test_equilibrium_criterion_with_wrong_series_name():
    available_series = {"mock_series": [1, 2, 3, 4, 5]}
    criterion = EquilibriumCriterion(series_name="test_series")
    with pytest.raises(AssertionError):
        criterion.in_equilibrium(available_series)


series = lambda values: {"test_series": values}


def test_equilibrium_criterion_not_in_equilibrium():
    criterion = EquilibriumCriterion("test_series", 10, 0.01)
    assert not criterion.in_equilibrium(series([]))
    assert not criterion.in_equilibrium(series([1]))
    assert not criterion.in_equilibrium(series([1, 2]))
    assert not criterion.in_equilibrium(series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    assert not criterion.in_equilibrium(series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))


def test_equilibrium_criterion_in_equilibrium():
    criterion = EquilibriumCriterion("test_series", 10, 0.01)
    balanced = criterion.in_equilibrium(
        series([1, 2.99, 3.00, 3.01, 3.00, 3.01, 3.01, 3.02, 3.03, 3.02, 3.01, 3.02])
    )
    assert balanced
