# Note: don't name this file just "jax", as it will conflict with the jax package.

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)