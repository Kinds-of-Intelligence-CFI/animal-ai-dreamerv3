from jax import random

key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)
