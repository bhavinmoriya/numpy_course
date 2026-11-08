"""Property-based tests: state a law, let hypothesis hunt for counterexamples."""
import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402
from hypothesis.extra import numpy as hnp  # noqa: E402

finite = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)


@given(hnp.arrays(np.float64, st.integers(0, 60), elements=finite))
def test_sort_is_sorted_permutation(x):
    s = np.sort(x)
    assert np.all(np.diff(s) >= 0)
    assert np.array_equal(np.sort(s), s) and len(s) == len(x)
    np.testing.assert_array_equal(np.sort(x), np.sort(x[np.random.default_rng(0).permutation(len(x))]))


# deadline=None: the first call imports the course module (which runs its demos once).
@settings(deadline=None)
@given(hnp.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=5),
       hnp.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=5))
def test_broadcast_shape_matches_numpy(s1, s2):
    from numpy_course.m02_broadcasting import broadcast_shape  # noqa: PLC0415

    try:
        expected = np.broadcast_shapes(s1, s2)
    except ValueError:
        with pytest.raises(ValueError):
            broadcast_shape(s1, s2)
    else:
        assert broadcast_shape(s1, s2) == expected


@given(hnp.arrays(np.float64, (4, 3), elements=st.floats(-10, 10)))
def test_svd_reconstructs(a):
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    np.testing.assert_allclose(u @ np.diag(s) @ vt, a, atol=1e-8)


@given(hnp.arrays(np.float64, st.integers(1, 40), elements=st.floats(-50, 50)))
def test_logsumexp_shift_invariance(x):
    lse = lambda v: v.max() + np.log(np.exp(v - v.max()).sum())  # noqa: E731
    np.testing.assert_allclose(lse(x + 1000.0), lse(x) + 1000.0, rtol=1e-10)
