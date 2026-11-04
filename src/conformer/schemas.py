import jax.numpy as jnp
from beartype import beartype
from flax.struct import PyTreeNode
from jaxtyping import Array, Bool, Float, Int, jaxtyped


@jaxtyped(typechecker=beartype)
class Feature3DWithLength(PyTreeNode):
    x: Float[Array, "B T D"]
    xlens: Int[Array, " B"]

    @property
    def valid_mask(self) -> Bool[Array, "B T"]:
        T = self.x.shape[1]
        return jnp.arange(T)[None, :] < self.xlens[:, None]
