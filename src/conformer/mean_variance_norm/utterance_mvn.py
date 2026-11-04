import jax.numpy as jnp
from flax import nnx

from conformer.schemas import Feature3DWithLength


class UtteranceMVN(nnx.Module):
    def __init__(self, norm_means: bool, norm_vars: bool, *, rngs: nnx.Rngs) -> None:
        """
        utterance-level mean and variance normalization.
        ref: https://github.com/espnet/espnet/blob/master/espnet2/layers/utterance_mvn.py
        """
        self.norm_means = norm_means
        self.norm_vars = norm_vars

    def __call__(self, feat: Feature3DWithLength) -> Feature3DWithLength:
        valid_mask = feat.valid_mask[:, :, None]  # (B, T, 1)

        x = jnp.where(valid_mask, feat.x, 0.0)
        xlens = feat.xlens[:, None, None]  # (B, 1, 1)

        # (B, 1, D)
        x_mean = (feat.x * valid_mask).sum(axis=1, keepdims=True) / xlens

        x_centered = jnp.where(valid_mask, x - x_mean, 0.0)

        if self.norm_means:
            x = x_centered
        if self.norm_vars:
            x_var = (x_centered**2).sum(axis=1, keepdims=True) / xlens
            x_var_safe = jnp.where(x_var > 0, x_var, 1.0)
            x_std = jnp.sqrt(x_var_safe)
            x = x / x_std

        new_x = jnp.where(valid_mask, x, 0.0)

        return feat.replace(x=new_x)
