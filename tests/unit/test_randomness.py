"""Named random domains are reproducible, independent, and order-free."""

from __future__ import annotations

import numpy as np

from src.core.utils.randomness import RandomDomains


def test_same_seed_and_name_reproduce_the_stream() -> None:
    first = RandomDomains(7).stream("training").normal(size=16)
    second = RandomDomains(7).stream("training").normal(size=16)
    np.testing.assert_array_equal(first, second)


def test_consecutive_draws_are_fresh() -> None:
    stream = RandomDomains(7).stream("training")
    assert not np.array_equal(stream.normal(size=16), stream.normal(size=16))


def test_streams_do_not_depend_on_creation_order() -> None:
    plain = RandomDomains(3)
    training = plain.stream("training").normal(size=16)

    with_diagnostics = RandomDomains(3)
    with_diagnostics.stream("diagnostics").normal(size=1000)
    np.testing.assert_array_equal(
        with_diagnostics.stream("training").normal(size=16), training)


def test_names_and_seeds_select_different_streams() -> None:
    domains = RandomDomains(3)
    assert not np.array_equal(
        domains.stream("a").normal(size=16), domains.stream("b").normal(size=16))
    assert not np.array_equal(
        RandomDomains(3).stream("a").normal(size=16),
        RandomDomains(4).stream("a").normal(size=16))


def test_replay_repeats_a_draw_without_touching_the_stream() -> None:
    domains = RandomDomains(5)
    first = domains.replay("finite-difference", replicate=2).normal(size=8)
    second = domains.replay("finite-difference", replicate=2).normal(size=8)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        domains.stream("finite-difference").normal(size=8),
        RandomDomains(5).stream("finite-difference").normal(size=8))


def test_integer_seed_is_stable_and_bounded() -> None:
    seed = RandomDomains(11).integer_seed("initialization")
    assert seed == RandomDomains(11).integer_seed("initialization")
    assert 0 <= seed < 2**63
