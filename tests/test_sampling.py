import random

import pytest

from autocheckout.sampling import largest_remainder, take_stratified


def test_largest_remainder_is_exact_and_within_quota():
    weights = {"a": 7, "b": 5, "c": 1, "d": 0}
    shares = largest_remainder(weights, 10)
    assert sum(shares.values()) == 10 and shares["d"] == 0
    for key, weight in weights.items():
        quota = 10 * weight / 13
        assert int(quota) <= shares[key] <= int(quota) + 1


def test_largest_remainder_breaks_ties_by_key():
    assert largest_remainder({"b": 1, "a": 1, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}
    with pytest.raises(ValueError):
        largest_remainder({"a": 0}, 3)


def test_take_stratified_reaches_targets_and_is_deterministic():
    units = [(f"u{i:02d}", "easy" if i % 2 else "hard", 1 + i % 3) for i in range(40)]
    chosen = take_stratified(units, {"easy": 10, "hard": 5}, random.Random(3))
    assert chosen == take_stratified(list(reversed(units)), {"easy": 10, "hard": 5}, random.Random(3))
    size = {key: (level, n) for key, level, n in units}
    for level, target in (("easy", 10), ("hard", 5)):
        taken = sum(size[k][1] for k in chosen if size[k][0] == level)
        assert target <= taken < target + 3  # overshoot is less than one unit
    assert chosen != take_stratified(units, {"easy": 10, "hard": 5}, random.Random(4))


def test_take_stratified_takes_everything_when_short():
    units = [("a", "easy", 2), ("b", "easy", 2)]
    assert take_stratified(units, {"easy": 10, "hard": 1}, random.Random(0)) == ["a", "b"]
