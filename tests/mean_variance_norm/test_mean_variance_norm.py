import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from conformer.mean_variance_norm.utterance_mvn import UtteranceMVN
from conformer.schemas import Feature3DWithLength

ALL_CONFIGS = [(True, False), (True, True), (False, True), (False, False)]


@pytest.fixture
def feat_test():
    """B=1, T=3, D=1, x = [1, 2, 3] -> mean = 2, var = 2/3"""
    return Feature3DWithLength(
        x=jnp.array([[[1.0], [2.0], [3.0]]]), xlens=jnp.array([3])
    )


def test_no_norm_returns_input(feat_test):
    out = UtteranceMVN(norm_means=False, norm_vars=False, rngs=nnx.Rngs(0))(feat_test)
    np.testing.assert_allclose(out.x, feat_test.x)


def test_means_only(feat_test):
    out = UtteranceMVN(norm_means=True, norm_vars=False, rngs=nnx.Rngs(0))(feat_test)
    np.testing.assert_allclose(out.x[0, :, 0], [-1.0, 0.0, 1.0], atol=1e-6)


def test_vars_only(feat_test):
    out = UtteranceMVN(norm_means=False, norm_vars=True, rngs=nnx.Rngs(0))(feat_test)
    np.testing.assert_allclose(
        out.x[0, :, 0], np.array([1.0, 2.0, 3.0]) / np.sqrt(2.0 / 3.0), atol=1e-5
    )


def test_means_and_vars(feat_test):
    out = UtteranceMVN(norm_means=True, norm_vars=True, rngs=nnx.Rngs(0))(feat_test)
    s = np.sqrt(1.5)  # 1 / sqrt(2/3)
    np.testing.assert_allclose(out.x[0, :, 0], [-s, 0.0, s], atol=1e-6)


@pytest.mark.parametrize("norm_means,norm_vars", ALL_CONFIGS)
def test_padding_is_ignored_and_zeroed(norm_means, norm_vars):
    mvn = UtteranceMVN(norm_means=norm_means, norm_vars=norm_vars, rngs=nnx.Rngs(0))
    x = jnp.array(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 40.0]],
            [[4.0, -1.0], [6.0, 1.0], [999.0, 999.0]],
        ]
    )
    out = mvn(Feature3DWithLength(x=x, xlens=jnp.array([3, 2])))
    alone = mvn(Feature3DWithLength(x=x[1:, :2], xlens=jnp.array([2])))

    np.testing.assert_allclose(out.x[1, :2], alone.x[0], atol=1e-5)
    np.testing.assert_array_equal(out.x[1, 2], [0.0, 0.0])
    np.testing.assert_array_equal(out.xlens, [3, 2])


def test_valid_frame_statistics():
    x = jax.random.normal(jax.random.key(0), (3, 7, 4)) * 5.0 + 3.0
    xlens = jnp.array([7, 5, 2])
    out = UtteranceMVN(norm_means=True, norm_vars=True, rngs=nnx.Rngs(0))(
        Feature3DWithLength(x=x, xlens=xlens)
    )
    for b, n in enumerate(xlens.tolist()):
        np.testing.assert_allclose(out.x[b, :n].mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(out.x[b, :n].var(axis=0), 1.0, atol=1e-4)


def test_constant_input_no_nan():
    mvn = UtteranceMVN(norm_means=True, norm_vars=True, rngs=nnx.Rngs(0))
    x = jnp.full((1, 4, 2), 5.0)
    xlens = jnp.array([4])

    out = mvn(Feature3DWithLength(x=x, xlens=xlens))
    np.testing.assert_allclose(out.x, 0.0)

    grad = jax.grad(lambda x: mvn(Feature3DWithLength(x=x, xlens=xlens)).x.sum())(x)
    assert jnp.all(jnp.isfinite(grad))


def test_jit_matches_eager():
    mvn = UtteranceMVN(norm_means=True, norm_vars=True, rngs=nnx.Rngs(0))
    feat = Feature3DWithLength(
        x=jax.random.normal(jax.random.key(1), (2, 5, 3)), xlens=jnp.array([5, 3])
    )
    np.testing.assert_allclose(jax.jit(mvn)(feat).x, mvn(feat).x, atol=1e-6)
