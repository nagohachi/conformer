import jax.numpy as jnp
from flax import nnx

from conformer.schemas import Feature3DWithLength


class UtteranceMVN(nnx.Module):
    def __init__(
        self, norm_means: bool, norm_vars: bool, eps: float = 1.0e-20, *, rngs: nnx.Rngs
    ) -> None:
        """
        utterance-level mean and variance normalization.
        ref: https://github.com/espnet/espnet/blob/master/espnet2/layers/utterance_mvn.py
        """
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.eps = eps

    def __call__(self, feat: Feature3DWithLength) -> Feature3DWithLength:
        valid_mask = feat.valid_mask[:, :, None]  # (B, T, 1)

        # (B, 1, D)
        hid_mean = (feat.x * valid_mask).sum(axis=1, keepdims=True) / feat.xlens[
            :, None, None
        ]

        mean_of_square = (feat.x**2 * valid_mask).sum(
            axis=1, keepdims=True
        ) / feat.xlens[:, None, None]
        hid_var = mean_of_square - hid_mean**2  # (B, 1, D)

        if self.norm_means:
            if self.norm_vars:
                new_x = (feat.x - hid_mean) / jnp.sqrt(hid_var)
            else:
                new_x = feat.x - hid_mean
        else:
            if self.norm_vars:
                new_x = feat.x / jnp.sqrt(hid_var)
            else:
                new_x = feat.x

        return feat.replace(x=new_x)
