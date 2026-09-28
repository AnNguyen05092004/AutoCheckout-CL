"""Deterministic apportionment and level-stratified sampling used by the data tools."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from fractions import Fraction
from typing import TypeVar

K = TypeVar("K", bound=Hashable)


def largest_remainder(weights: Mapping[K, int], total: int) -> dict[K, int]:
    """Split ``total`` into integers proportional to ``weights`` (Hamilton's method).

    Each share is the floor or the ceiling of its exact quota and the shares sum to ``total``.
    Equal remainders are broken by key order, so the result is deterministic.
    """
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("weights must have a positive sum")
    quotas = {key: Fraction(total * weight, weight_sum) for key, weight in weights.items()}
    shares = {key: math.floor(quota) for key, quota in quotas.items()}
    missing = total - sum(shares.values())
    by_remainder = sorted(quotas, key=lambda key: (-(quotas[key] - shares[key]), key))
    for key in by_remainder[:missing]:
        shares[key] += 1
    return shares


def take_stratified(units: Iterable[tuple[K, str, int]], targets: Mapping[str, int],
                    rng: random.Random) -> list[K]:
    """Draw ``(key, stratum, size)`` units until each stratum's drawn size reaches its target.

    Units of a stratum are sorted, shuffled with ``rng`` and taken in that order while the
    stratum total is below its target, so a stratum overshoots by less than one unit's size.
    Strata are processed in sorted order. Returns the chosen keys, sorted.
    """
    pools: dict[str, list[tuple[K, int]]] = defaultdict(list)
    for key, stratum, size in units:
        pools[stratum].append((key, size))
    chosen: list[K] = []
    for stratum in sorted(targets):
        pool = sorted(pools.get(stratum, []))
        rng.shuffle(pool)
        taken = 0
        for key, size in pool:
            if taken >= targets[stratum]:
                break
            chosen.append(key)
            taken += size
    return sorted(chosen)
