from computational_models.core.equilibrium_criterion import EquilibriumCriterion


def test_equilibrium_criterion_with_default_values():
    criterion = EquilibriumCriterion()
    assert criterion.window_size == 20
    assert criterion.tolerance == 0.001


def test_equilibrium_criterion_with_custom_values():
    criterion = EquilibriumCriterion(10, 0.01)
    assert criterion.window_size == 10
    assert criterion.tolerance == 0.01


def test_equilibrium_criterion_not_in_equilibrium():
    criterion = EquilibriumCriterion(10, 0.01)
    assert not criterion.in_equilibrium([])
    assert not criterion.in_equilibrium([1])
    assert not criterion.in_equilibrium([1, 2])
    assert not criterion.in_equilibrium([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert not criterion.in_equilibrium([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])


def test_equilibrium_criterion_in_equilibrium():
    criterion = EquilibriumCriterion(10, 0.01)
    balanced = criterion.in_equilibrium(
        [1, 2.99, 3.00, 3.01, 3.00, 3.01, 3.01, 3.02, 3.03, 3.02, 3.01, 3.02]
    )
    assert balanced
